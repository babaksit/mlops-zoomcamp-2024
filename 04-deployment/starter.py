#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import argparse


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def get_model_and_dv():
    model_path = 'model.bin'
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def get_yellow_taxi_df(year, month):
    data_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = read_data(data_path)
    return df


def predict(df, dv, model):
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def pipeline(year, month):
    dv, model = get_model_and_dv()
    df = get_yellow_taxi_df(year, month)
    y_pred = predict(df, dv, model)
    mean_pred_duration = y_pred.mean()
    print(f"Mean predicted duration: {mean_pred_duration:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Year of the yellow taxi data')
    parser.add_argument('--month', type=int, help='Month of the yellow taxi data')
    args = parser.parse_args()

    pipeline(args.year, args.month)
