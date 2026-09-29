from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import DATA_DIR, EXPECTED_SCHEMA, OUTPUT_DIR, SEED, VARIANT_COLUMNS, prepare_output_dir, set_seeds
from src.image_proc import run_image_tasks
from src.io_utils import check_data_quality, extract_columns, read_table, show_table, write_main_outputs
from src.matrix_ops import invert_random_matrix, make_large_sparse, pca_pipeline_search
from src.plotting import convolutions, cross_correlations, derivatives_gradients, plot_ecdf, plot_fft, plot_histograms, plot_interpolation, plot_periodograms, plot_source, plot_spectrograms, plot_statistics
from src.preprocessing import automatic_cleaning, cast_metadata_types, interactive_cleaning_menu, moving_average, sliding_window, split_train_val_test, to_numpy
from src.scaling_tasks import binary_masks, build_feature_tensor, compare_scalers
from src.stats_utils import column_statistics, confidence_intervals, correlation_significance, covariance_correlation, distribution_tests, norms, sorted_columns, vector_products
from src.tensor_ops import create_tpq, keras_expression, pytorch_tasks, tensorflow_tasks, torch_expression


def section(number: str, title: str):
    print(f"\n{'='*78}\nЗадание {number}. {title}\n{'='*78}")


def main(interactive_cleaning: bool = False):
    warnings.simplefilter("always")
    prepare_output_dir()
    print("Лабораторная работа №1 — вариант 4")
    print("Датасет: data/dataset_var4.csv")
    print("Столбцы варианта:", VARIANT_COLUMNS)

    section("13","Фиксация seed"); print(json.dumps(set_seeds(SEED),ensure_ascii=False,indent=2))

    section("1/4/5","Чтение CSV, проверка качества и вывод таблицы")
    raw=read_table(DATA_DIR/"dataset_var4.csv"); show_table(raw,10); report=check_data_quality(raw,EXPECTED_SCHEMA); print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

    section("6/7/8","Столбцы варианта, явные типы, обработка проблем")
    typed=cast_metadata_types(raw); selected=extract_columns(typed,VARIANT_COLUMNS)
    if interactive_cleaning:
        clean=interactive_cleaning_menu(selected,VARIANT_COLUMNS); actions={"mode":"interactive"}
    else:
        clean,actions=automatic_cleaning(selected,VARIANT_COLUMNS); actions["mode"]="automatic mean fill"
    clean=clean.astype({c:"float64" for c in VARIANT_COLUMNS}); print(clean.dtypes); print("Действия обработки:",actions)

    section("9","DataFrame -> NumPy"); x=to_numpy(clean); print("X shape:",x.shape,"dtype:",x.dtype)

    section("10/11/34/35","Масштабирование, sklearn и обратное преобразование")
    scaling_cmp,inverse_cmp,_=compare_scalers(x,VARIANT_COLUMNS); print(scaling_cmp.to_string(index=False)); print(inverse_cmp.to_string(index=False))

    section("2","Запись XLSX/CSV/TXT/MAT")
    written=write_main_outputs(clean,scaling_cmp,inverse_cmp,"variant4"); print(json.dumps(written,ensure_ascii=False,indent=2))
    # Проверка чтения четырёх форматов.
    for fmt,path in written.items():
        rt=read_table(path); print(f"{fmt}: shape={rt.shape}, columns={rt.columns.tolist()}")

    section("12","Разбиение train/val/test"); tr,va,te=split_train_val_test(x,(70,15,15)); print("train/val/test:",tr.shape,va.shape,te.shape)

    section("3/14/15","PNG-функция, исходный график и гистограммы"); plot_source(clean); plot_histograms(clean); print("Графики построены. В output сохраняются только PNG задания 50.")

    section("16","Отсортированные столбцы")
    for col,arr in sorted_columns(clean).items(): print(f"\n{col}:\n{arr}")

    section("17","Эмпирическая функция распределения"); plot_ecdf(clean); print("График построен.")

    section("18","Среднее, выборочная дисперсия, мода, медиана"); stats_df=column_statistics(clean); print(stats_df); plot_statistics(stats_df)

    section("19","95% доверительные интервалы среднего и дисперсии"); print(confidence_intervals(clean,.05))

    section("20","Ковариация и корреляция"); cov_np,corr_np,cov_pd,corr_pd=covariance_correlation(clean); print("Covariance NumPy:\n",cov_np); print("Correlation NumPy:\n",corr_np)

    section("21","Значимость коэффициентов корреляции"); print(correlation_significance(clean).to_string(index=False))

    section("22","Взаимная корреляция"); cross_correlations(clean); print("Вычислено и построено.")

    section("23","Численная производная и градиент"); derivatives_gradients(clean); print("Вычислено и построено.")

    section("24","Свёртка столбцов"); conv=convolutions(clean); print({k:v.shape for k,v in conv.items()})

    section("25","Скалярное и векторное произведения"); print(vector_products(clean).to_string(index=False))

    section("26","L1/L2 нормы"); print(norms(clean))

    section("27","Гипотезы о равномерном и нормальном распределении"); print(json.dumps(distribution_tests(clean,.05),ensure_ascii=False,indent=2,default=str))

    section("28","Спектрограммы"); plot_spectrograms(clean); print("Построено.")
    section("29","Периодограммы"); plot_periodograms(clean); print("Построено.")
    section("30","АЧХ через FFT"); plot_fft(clean); print("Построено.")
    section("31","Кубическая интерполяция"); plot_interpolation(clean,"cubic"); print("Построено.")
    section("32","Сплайн-интерполяция"); plot_interpolation(clean,"spline"); print("Построено.")

    section("33","Бинарные маски"); masks=binary_masks(x); print(json.dumps({k:np.sum(v,axis=0).tolist() for k,v in masks.items()},ensure_ascii=False,indent=2))

    section("36/37","Скользящее окно и скользящее среднее"); win=sliding_window(x[:,0],10); ma=moving_average(x[:,0],10); print("window shape:",win.shape,"moving average shape:",ma.shape)

    section("38-45","TensorFlow / PyTorch / Keras")
    try:
        tf_res=tensorflow_tasks(x,SEED); print("TensorFlow result shape:",tf_res["result"].shape)
    except Exception as exc: print("TensorFlow недоступен в текущем окружении:",exc)
    try:
        pt_res=pytorch_tasks(x,SEED); print("PyTorch result shape:",pt_res["result"].shape)
    except Exception as exc: print("PyTorch недоступен:",exc)
    t,p,q=create_tpq(SEED)
    vk=None
    try:
        vk=keras_expression(t,p,q); print("Keras V shape:",vk.shape)
    except Exception as exc: print("Keras недоступен:",exc)
    try:
        vt=torch_expression(t,p,q); print("PyTorch V shape:",vt.shape)
        if vk is not None: print("Keras/PyTorch max abs diff:",float(np.max(np.abs(vk-vt))))
    except Exception as exc: print("PyTorch expression error:",exc)

    section("46","Разреженный массив 100000x100000"); s=make_large_sparse(SEED); print({"shape":s.shape,"nnz":s.nnz,"format":s.getformat()})
    section("47","Обратная матрица 100x100"); inv_ok,err=invert_random_matrix(SEED); print("invertible:",inv_ok,"identity error:",err)

    section("48","Многомерный массив F"); f,names,shares=build_feature_tensor(x); print("F shape:",f.shape,"channels:",names,"source X_view shares memory:",shares)

    section("49","PCA pipeline: scaler + признаки F"); pca_results,best=pca_pipeline_search(f,names,SEED); print(pca_results.to_string(index=False)); print("Выбранная комбинация:",best)

    section("50","Обработка цветного изображения"); image_res=run_image_tasks(); print(json.dumps({k:v for k,v in image_res.items() if k!="plots"},ensure_ascii=False,indent=2,default=str))

    print("\nГотово. Все результаты сохранены в:",OUTPUT_DIR)
    files=sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())
    print(f"В output файлов: {len(files)}")
    for name in files: print(" -",name)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Лабораторная работа №1, вариант 4")
    parser.add_argument("--interactive-cleaning",action="store_true")
    args=parser.parse_args(); main(args.interactive_cleaning)
