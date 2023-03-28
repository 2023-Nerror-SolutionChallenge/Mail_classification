from flask_restful import reqparse
from flask import Flask, jsonify,request

import numpy as np
import pickle as p
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import pymysql

app = Flask(__name__)

@app.route('/smart-scan', methods=['GET'])
def smart_scan():
    username = request.args.get('username')
    content = db_connector(username)

    removable_idx = []
    result = predict(content)
    return jsonify(result)


def predict(contents):
    df=pd.read_csv('./maildataset.csv',encoding='latin-1')
        
    df1=df.where(pd.notnull(df),'')
    df1["v1"].replace('ham',1,inplace=True)
    df1["v1"].replace('spam',0,inplace=True)

    x=df1["v2"]
    y=df1["v1"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

    feature_extraction= TfidfVectorizer(min_df=1,stop_words='english',lowercase= True)
    x_train_feature=feature_extraction.fit_transform(x_train)
    x_test_feature=feature_extraction.transform(x_test)

    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    model_from_joblib = joblib.load('./mail_model.pkl')
    model_from_joblib.fit(x_train_feature,y_train)

    # 메일 내용 -> 모델 예측 요청
    removable_idx = []
    for content in contents:
        input_mail= [content[2]]

        # 벡터로 변환
        input_data_features = feature_extraction.transform(input_mail)

        # 예측
        prediction = model_from_joblib.predict(input_data_features)

        if (prediction[0]==1):
            print("정상 메일로 분류되어 삭제하지 않습니다")
        else:
            # 삭제할 mail_id 를 추가
            removable_idx.append(content[0])
            print("\n\n\n스팸 메일로 분류되었습니다")

    return removable_idx


def db_connector(username):
    db = {

        'user': 'root',
        'password': 'qwerty1234',
        'host': '34.22.71.172',
        'port': 3306,
        'database': 'marbon'
    }
    db_connection = pymysql.connect(host='34.22.71.172',
                                    port=3306,
                                    user='root',
                                    password='qwerty1234',
                                    charset='utf8mb4')
    cursor = db_connection.cursor()
    sql = "SELECT * FROM marbon.mail_data where username=%s"

    cursor.execute(sql, username)
    return cursor.fetchall()

if __name__=="__app__":
    app.run(debug=True)