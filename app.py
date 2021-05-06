from flask import Flask, render_template, make_response, request, jsonify
from flask import abort, redirect, url_for
from jinja2 import Template

import urllib.request
import requests

import os
import json
import werkzeug
import zipfile
import shutil
import time
import subprocess
import pickle
from datetime import datetime

from tensorflow_federated.python.core.backends.native import execution_contexts
import my_fl_train
import my_fl_pred

app = Flask(__name__)

# FL-Client API

# limited upload file size: 100MB
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 100


# stylesheet更新用の関数（システムとは無関係）
@app.context_processor
def add_staticfile():
    def staticfile_cp(fname):
        path = os.path.join(app.root_path, 'static/css', fname)
        mtime =  str(int(os.stat(path).st_mtime))
        return '/static/css/' + fname + '?v=' + str(mtime)
    return dict(staticfile=staticfile_cp)

def terminal_interface(cmd):
    """
    :param cmd: str 実行するコマンド
    """
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        line = proc.stdout.readline()
        if line:
            yield line
        if not line and proc.poll() is not None:
            break

    return line


def input_command(cmd):
    # コマンドを実行する関数
    # 注：subprocess.check_output ではなぜか標準出力が取得できなかったので，
    # 以下の方法を試したらうまくいった．
    _output = []  # 標準出力を格納
    for line in terminal_interface(cmd):
        _output.append(line.decode())
    return _output


@app.route('/rqp-web-client')
def rqp_web_client():
    if request.args.get('access_token'):
        rpt = request.args.get('access_token')
        html = """
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
        Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml">

        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css"
                href="/static/css/style.css">
            <link rel="stylesheet" type="text/css"
                href="/static/css/procedure.css">
            <title>FL-Client</title>
        </head>

        <body>
            <h1>FL-Client</h1>
            <p>Request a model delta from the FL-Server.</p><br>
            <form action="/req-resource" method="post">
                <p>FL-Client User ID:
                    <input type="text" name="uid"></p>
                <p>Resource ID:
                    <input type="text" name="rid"></p>
                <input type="hidden" name="rpt" value={0}>
                <button type="submit" value="request">Request</button>
            </form>
            <br>
            <br>
            <blockquote>
            <u>Procedure 12</u><br>
            The FL-Cient again requests the FL-Server to execute <i>tff</i> on the resource. (14)<br>
            The FL-Server receives the RPT from the client and retrieves the information associated with the RPT from the Authorization Blockchain. (15)<br>
            If the RPT is <i>active</i> and the allowed scopes include <i>tff</i>, the FL-Server will allow the FL-Client to execute <i>tff</i> on the resource.<br>
            The FL-Client gets the model deltas that are allowed to execute <i>tff</i>. (16)<br>
            </blockquote> 

            <p><img src="/static/images/access01.png" width="673" height="400"></p>

        </body>

        </html>
        """.format(rpt)
    else:
        html = """
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
        Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml">

        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css"
                href="/static/css/style.css">
            <link rel="stylesheet" type="text/css"
                href="/static/css/procedure.css">
            <title>FL-Client</title>
        </head>

        <body>
            <h1>FL-Client</h1>
            <p>Request a model delta from the FL-Server.</p><br>

            <form action="/req-resource" method="post">
                <p>FL-Client User ID:
                    <input type="text" name="uid"></p>
                <p>Resource ID:
                    <input type="text" name="rid"></p>
                <button type="submit" value="request">Request</button>
            </form>
            <br>
            <br>
            <blockquote>
            <u>Procedure 08</u><br>
            The FL-Client requests the FL-Server to execute <i>tff</i> on the resource. (6)<br>
            The FL-Server receives this request, and then requests the Authorization Blockchain to perform the authorization process. (7)<br>
            The Authorization Blockchain receives a request and starts the authorization process.<br>
            </blockquote>

            <p><img src="/static/images/authz01.png" width="673" height="400"></p>

        </body>

        </html>
        """
    template = Template(html)
    return template.render()


@app.route('/req-resource', methods=['post'])
def req_resource_post():
    # RS と通信する
    """
    :req_param user_id: RqP のユーザ ID
    :req_param resource_id:
    """
    uid = request.form['uid']
    if request.form['rid'] != "":
        rid = request.form['rid']
    else:
        return make_response(jsonify({'error': "no rid"}), 403)

    try:
        rpt = request.form['rpt']
    except:
        rpt = ""

    # リソースを RS に要求するリクエストを生成する(fl-server.ctiport.net:8080/resource)
    req_resource_url = 'http://fl-server.ctiport.net:8080/resource'
    data = {
        'resource_id': rid,
        'request_scopes': ['tff']
    }
    flag = False
    if rpt != "":
        flag = True

    if flag:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': rpt
        }
    else:
        headers = {
            'Content-Type': 'application/json'
        }
    req = urllib.request.Request(url=req_resource_url, data=json.dumps(
        data).encode('utf8'), headers=headers)

    # RPT を含めた場合
    if flag:
        DIR = './zipped/'
        DOWNLOAD_SAVE_DIR = DIR + uid
        try:
            # リソース（ファイル）をダウンロードする
            os.makedirs(DOWNLOAD_SAVE_DIR, exist_ok=True)
            response = urllib.request.urlopen(req)

            contentType = response.headers['Content-Type']
            contentDisposition = response.headers['Content-Disposition']
            ATTRIBUTE = 'filename='
            fileName = contentDisposition[contentDisposition.find(
                ATTRIBUTE) + len(ATTRIBUTE):]
            print(contentDisposition)
            print(fileName)

            saveFileName = fileName
            saveFilePath = os.path.join(DOWNLOAD_SAVE_DIR, saveFileName)
            with open(saveFilePath, mode='wb') as fp:
                fp.write(response.read())

        except Exception as e:
            print(e)
            return make_response(jsonify({'response': "error: invalid user id"}), 400)

        # 他にリソースを要求するか，予測アプリを使うかを選択させる html を表示する．uid は引き継ぐ
        html = """
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0
        Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml">

        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css"
                href="/static/css/style.css">
            <link rel="stylesheet" type="text/css"
                href="/static/css/procedure.css">
            <title>FL-Client</title>
        </head>

        <body>
            <h1>FL-Client</h1>
            <p><b>FL-Client ID: {0}</b></p>
            <p>Successfully retrieved the resource.</p>
            <h2>List of acquired resources</h2>
            <ul>""".format(uid)
        
        resources = os.listdir(DOWNLOAD_SAVE_DIR)

        # FL-Serverとの間でROのユーザIDを共有する仕組みはないので，
        # デモ用にROのユーザIDを表示する仕組みを無理やり実装
        for e in resources:
            _e = e.replace(".zip", "")
            if _e == "08db20ba-2666-5b91-9bef-3d5b7d9138ae":
                _owner = "  from  <b>user01</b>"
            elif _e == "1c1f1d9f-051c-592f-bb06-5ec8cef664ba":
                _owner = "  from  <b>user02</b>"
            elif _e == "7b7f4414-a949-5e48-a669-2f203efe6e3f":
                _owner = "  from  <b>user03</b>"
            else:
                _owner = ""
            
            html += "<li>" + "<b>" + _e + "</b>" + _owner + "</li>"

        html += """
            </ul>
            <p>Press "Continue" to continue with the resource acquisition, or "Predict" to move on to the prediction.</p>
            <br>
            <form action="/rqp-web-client" method="get">
                <button type="submit" value="continue">Continue</button>
                <input type="hidden" name="uid" value={0}>
            </form>
            <form action="/application" method="post">
                <button type="submit" value="predict">Predict</button>
                <input type="hidden" name="uid" value={0}>
            </form>
            <br>
            <br>
            <blockquote>
            <u>Procedure 13</u><br>
            Choose whether to continue with the request of another resource or to execute the application.<br>
            </blockquote>

            <p><img src="/static/images/access02.png" width="673" height="400"></p>
        </body>

        </html>
        """.format(uid)

        template = Template(html)
        return template.render()

    # RPT を含めなかった場合
    else:
        # Request to http://fl-server.ctiport.net:8080/resource
        with urllib.request.urlopen(req) as res:
            body = res.read()
            body = body.decode('utf8').replace("'", '"')
            body = json.loads(body)
        ticket = body['response']['ticket']
        token_endpoint = body['response']['token_endpoint']

        # トークンエンドポイント(/token)へ誘導する
        html = """
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
        Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml">

        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css"
                href="/static/css/style.css">
            <link rel="stylesheet" type="text/css"
                href="/static/css/procedure.css">
            <title>FL-Client</title>
        </head>

        <body>
            <h1>FL-Client</h1>
            <p><b>FL-Client ID: {3}</b></p>
            <p>Start the authorization process.</p>
            <p>Requesting resource <b>{0}</b>.</p>
            <br>
            <h2>Start communication with the token endpoint.</h2>
            <form action="/req-token" method="post">
                <button type="submit" value="authorize">Request permission</button>
                <input type="hidden" name="ticket" value={1}>
                <input type="hidden" name="token_endpoint" value={2}>
            </form>
            <br>
            <br>
            <blockquote>
            <u>Procedure 09</u><br>
            The FL-Client requests permission from the Authorization Blockchain. (8)<br>
            The Authorization Blockchain checks that the request content does not exceed the scope allowed for the resource.<br>
            </blockquote>

            <p><img src="/static/images/authz02.png" width="673" height="400"></p>
        </body>

        </html>
        """.format(rid, ticket, token_endpoint, uid)

        template = Template(html)

        return template.render()


@app.route('/req-token', methods=['post'])
def req_token():
    ticket = request.form['ticket']
    token_endpoint = request.form['token_endpoint']

    timestamp = "1595230979"
    timeSig = "vF9Oyfm+G9qS4/Qfns5MgSZNYjOPlAIZVECh2I5Z7HHgdloy5q7gJoxi7c1S2/ebIQbEMLS05x3+b0WD0VJfcWSUwZMHr3jfXYYwbeZ1TerKpvfp1j21nZ+OEP26bc28rLRAYZsVQ4Ilx7qp+uLfxu9X9x37Qj3n0CI2TEiKYSSYDQ0bftQ/3iWSSoGjsDljh9bKz1eVL911KeUGO+t/9IkB6LtZghdbIlnGISbgrVGoEOtGHi0t8uD2Vh/CRyBe+XnQV3HQtkjddLQitAesKTYunK1Ctia3x7klVjRH9XiJ11q6IbR8gz7rchdHYZe6HP+w/LyWMS5z6M26AXQrVw=="
    grant_type = "urn:ietf:params:oauth:grant-type:uma-ticket"

    data = {
        'grant_type': grant_type,
        'ticket': ticket,
        'timestamp': timestamp,
        'timeSig': timeSig
    }
    headers = {
        'Content-Type': 'application/json'
    }
    token_req = urllib.request.Request(
        url=token_endpoint, data=json.dumps(data).encode('utf8'), headers=headers)

    # Request to http://authz-blockchain.ctiport.net:8888/token
    with urllib.request.urlopen(token_req) as res:
        body = res.read()
        body = body.decode('utf8').replace("'", '"')
        body = json.loads(body)
        err_msg = body['response']['Error']
        if err_msg != "need_info":
            return make_response(jsonify({'error': "invalid phase"}))

        ticket = body['response']['Ticket']
        redirect_user = body['response']['RedirectUser']  # クレームの取得先 URI
        # state = body['response']['State'] # セキュリティ上，stateがあれば望ましいが本実装では省略

    # Authorization Blockchain に登録した client_id をパラメータに含める
    client_id = "client_id"
    # claims_redirect_uri を設定しておく（claim_token を受け取るための URI）
    claims_redirect_uri = "http://fl-client.ctiport.net:8888/redirect-claims"

    param = {
        'client_id': client_id,
        'ticket': ticket,
        'claims_redirect_uri': claims_redirect_uri,
        'timestamp': timestamp,
        'timeSig': timeSig
    }
    qs = urllib.parse.urlencode(param)

    return redirect(redirect_user + '?' + qs, code=301)


@app.route('/redirect-claims')
def redirect_claims():
    # 認証サーバからクレームトークンを受け取る用エンドポイント
    # パラメータを受け取る
    uid = request.args.get('uid')
    ticket = request.args.get('ticket')
    claim_token = request.args.get('claim_token')
    token_endpoint = request.args.get('token_endpoint')

    # Client を再度トークンエンドポイント(/token)へ誘導する
    html = """
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
    Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml">

    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" type="text/css"
            href="/static/css/style.css">
        <link rel="stylesheet" type="text/css"
            href="/static/css/procedure.css">
        <title>FL-Client</title>
    </head>

    <body>
        <h1>FL-Client</h1>
        <p><b>FL-Client ID: {3}</b></p>
        <p>Perform the authorization process.</p>
        <p><b>The claim token has been obtained.</b></p>
        <br>
        <h2>Start communication with the token endpoint.</h2>
        <form action="/req-token-after-claims" method="post">
            <button type="submit" value="authorize">Request permission</button>
            <input type="hidden" name="ticket" value={0}>
            <input type="hidden" name="token_endpoint" value={1}>
            <input type="hidden" name="claim_token" value={2}>
        </form>
        <br>
        <br>
        <blockquote>
        <u>Procedure 11</u><br>
        The FL-Client requests permission from the Authorization Blockchain again. (12)<br>
        The FL-Client sends the claim token at the same time.
        The Authorization Blockchain verifies that the received claim token satisfies the authorization policy associated with the resource.
        If the verification is successful, an access token, called RPT, is issued to the FL-Client. (13)<br>
        </blockquote>

        <p><img src="/static/images/authz04.png" width="673" height="400"></p>
    </body>

    </html>
    """.format(ticket, token_endpoint, claim_token, uid)

    template = Template(html)

    return template.render()


@app.route('/req-token-after-claims', methods=['post'])
def redirect_token_after_claims():
    # パラメータを受けとる
    ticket = request.form['ticket']
    token_endpoint = request.form['token_endpoint']
    claim_token = request.form['claim_token']

    timestamp = "1595230979"
    timeSig = "vF9Oyfm+G9qS4/Qfns5MgSZNYjOPlAIZVECh2I5Z7HHgdloy5q7gJoxi7c1S2/ebIQbEMLS05x3+b0WD0VJfcWSUwZMHr3jfXYYwbeZ1TerKpvfp1j21nZ+OEP26bc28rLRAYZsVQ4Ilx7qp+uLfxu9X9x37Qj3n0CI2TEiKYSSYDQ0bftQ/3iWSSoGjsDljh9bKz1eVL911KeUGO+t/9IkB6LtZghdbIlnGISbgrVGoEOtGHi0t8uD2Vh/CRyBe+XnQV3HQtkjddLQitAesKTYunK1Ctia3x7klVjRH9XiJ11q6IbR8gz7rchdHYZe6HP+w/LyWMS5z6M26AXQrVw=="
    grant_type = "urn:ietf:params:oauth:grant-type:uma-ticket"
    claim_token_format = "http://openid.net/specs/openid-connect-core-1_0.html#IDToken"

    data = {
        'grant_type': grant_type,
        'ticket': ticket,
        'claim_token': claim_token,
        'claim_token_format': claim_token_format,
        'timestamp': timestamp,
        'timeSig': timeSig
    }
    headers = {
        'Content-Type': 'application/json'
    }
    req = urllib.request.Request(url=token_endpoint, data=json.dumps(
        data).encode("utf-8"), headers=headers)

    # /token へリクエストを投げてレスポンスを得る
    with urllib.request.urlopen(req) as res:
        body = res.read()
        body = body.decode('utf8').replace("'", '"')
        body = json.loads(body)
        try:
            rpt = body['response']['token']  # rpt ない場合の処理も書く必要あり
        except:
            return make_response(jsonify({'error': "rpt may not be issued."}), 400)

    param = {
        'access_token': rpt
    }
    qs = urllib.parse.urlencode(param)

    return redirect(url_for('rqp_web_client') + '?' + qs, code=301)


@app.route('/application', methods=['post'])
def application():
    uid = request.form['uid']
    # RqP のデータアップロード用 html を投げる
    html = """
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
    Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" type="text/css"
            href="/static/css/style.css">
        <link rel="stylesheet" type="text/css"
            href="/static/css/procedure.css">
        <title>FL-Client</title>
    </head>

    <body>
        <h1>FL-Client</h1>
        <p><b>FL-Client ID: {0}</b></p>
        <p>Requesting party inputs unlabeled data to the client and asks it to predict the labels.</p>
        <br>
        <h2>Select and upload your data.</h2>
        <form action="/prediction" method="post" enctype="multipart/form-data">
            <input type="file" name="uploadFile" accept=".zip" required>
            <br>
            <button type="submit" value="predict">Predict</button>
            <input type="hidden" name="uid" value={0}>
        </form>
        <br>
        <br>
        <blockquote>
        <u>Procedure 14</u><br>
        The client constructs a machine learning model by averaging the model differences obtained in the previous phase. (17)<br>
        The requesting party gives unlabeled data to the client as input. (18)<br>
        The client performs label prediction using the unlabeled data and the machine learning model.<br>
        </blockquote>

        <p><img src="/static/images/app01.png" width="673" height="400"></p>
    </body>

    </html>
    """.format(uid)

    template = Template(html)

    return template.render()


@app.route('/prediction', methods=['post'])
def prediction():

    uid = request.form['uid']

    # RqP がアップロードしたファイル(zip)を確認
    if 'uploadFile' not in request.files:
        return make_response(jsonify({'response': "error: upload file is required"}), 400)
    file = request.files['uploadFile']
    filename = file.filename
    if '' == filename:
        return make_response(jsonify({'response': "file name must not be empty"}), 400)

    # zip ファイルを保存
    UPLOAD_PARENT = "./uploaded/"
    UPLOAD_DIR = UPLOAD_PARENT + uid + '/'
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_") \
            + werkzeug.utils.secure_filename(filename)
        file.save(os.path.join(UPLOAD_DIR, saveFileName))
        # zip を展開
        with zipfile.ZipFile(UPLOAD_DIR + saveFileName) as existing_zip:
            existing_zip.extractall(UPLOAD_DIR)
        # zip を削除
        os.remove(UPLOAD_DIR + saveFileName)
    except:
        return make_response(jsonify({'response': "error: invalid user id"}), 400)

    # ROのリソース（データセット）からモデルを構築
    # zip ファイルからデータセットを取り出す (from ./zipped to ./unzipped)
    RESOURCE_PARENT = "./zipped/"
    RESOURCE_DIR = RESOURCE_PARENT + uid + '/'
    UNZIP_PARENT = "./unzipped/"
    UNZIP_DIR = UNZIP_PARENT + uid + '/'
    li_resource = os.listdir(RESOURCE_DIR)
    print("li_resource: ", li_resource)
    TMP_DIR = "./.tmp/"
    for resource in li_resource:
        # リソースの zip ファイルを解凍 in ./.tmp
        TMP_UNZIP = TMP_DIR + resource.replace('.zip', '') + '/'
        with zipfile.ZipFile(RESOURCE_DIR + resource) as existing_zip:
            existing_zip.extractall(TMP_UNZIP)
        li_zipped_dataset = os.listdir(TMP_UNZIP)
        print("li_zipped_dataset: ", li_zipped_dataset)
        for zipped_dataset in li_zipped_dataset:
            # データセットの zip ファイルを解凍 in ./.tmp
            TMP_DATASET_DIR = TMP_DIR + zipped_dataset.replace('.zip', '') + '/'
            with zipfile.ZipFile(TMP_UNZIP + zipped_dataset) as existing_zip:
                existing_zip.extractall(TMP_DATASET_DIR)
            li_datadir = os.listdir(TMP_DATASET_DIR)
            print("li_datadir: ", li_datadir)
            for datadir in li_datadir:
                # ラベル付きデータの各ディレクトリを参照
                TMP_DATA_DIR = TMP_DATASET_DIR + datadir
                # ディレクトリを移動する．既にある場合は上書き (./unzipped/$rqp_id/$label/)
                if os.path.exists(UNZIP_DIR + datadir):
                    shutil.rmtree(UNZIP_DIR + datadir)
                shutil.move(TMP_DATA_DIR, UNZIP_DIR + datadir)

    # RO のデータセットを用いてモデルを構築する
    li_datadir = os.listdir(UNZIP_DIR)
    execution_contexts.set_local_execution_context()  # backend で tff を実行するために必要？
    #model = my_fl_train.federated_train(UNZIP_DIR, li_datadir)  # エラー
    MODEL_PARENT = './model/'
    MODEL_DIR = MODEL_PARENT + uid + '/'
    MODEL_NAME = 'model_for_' + uid
    os.makedirs(MODEL_DIR, exist_ok=True)

    # モデル作成
    cmd = "python3 my_fl_train.py " + UNZIP_DIR + " " + ",".join(li_datadir) + " " + MODEL_DIR + " " + MODEL_NAME
    print(cmd)
    li_output = input_command(cmd)
    for e in li_output:
        print(e)
    output = li_output[-1].strip()
    print(output)
    if output != "success":
        return "Error"
    

    with open(MODEL_DIR + MODEL_NAME, "rb") as f:
        model = pickle.load(f)

    # モデルと RqP のデータセットを用いて予測する
    MODE = 0  # ここで操作する
    if MODE == 0:  # ラベル無しデータセットのラベル予測
        result = my_fl_pred.federated_eval(model, UPLOAD_DIR, mode=0)
        result = result[90:100]  # 見栄え上，いい感じのとこだけ抜粋
        #return render_template('prediction.html', uid=uid, label=str(result))
        html = """
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
        Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        <html xmlns="http://www.w3.org/1999/xhtml">

        <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" type="text/css"
            href="/static/css/style.css">
        <link rel="stylesheet" type="text/css"
            href="/static/css/procedure.css">
        <title>FL-Client</title>
        </head>

        <body>
            <h1>FL-Client</h1>
            <p><b>FL-Client ID: {}</b></p>
            <br>
            <p>The machine learning model used for prediction is built from the resources retrieved by the FL-Client from the FL-Server.</p>
            <p>Predicted labels given by the model to the dataset uploaded by the FL-Requestor is as follows.</p>
            <div class="box11">
            <p>Result:</p>
            <table>
                <tr>
                    <th>index</th><td>predicted label</td>
        """.format(uid)
        tmp = ""
        for i, label in enumerate(result):
            tmp += "<tr><th>"
            tmp += str(i)
            tmp += "</th><td>"
            tmp += str(label)
            tmp += "</td></tr>"
        html += tmp

        html += """
            </table>
            </div>
            <blockquote>
            <u>Procedure 15</u><br>
            The label prediction results for the input data are displayed. (19)<br>
            </blockquote>

            <p><img src="/static/images/app02.png" width="673" height="400"></p>
        </body>

        </html>
        """
        template = Template(html)
        return template.render()

    elif MODE == 1:  # ラベルありデータセットの予測精度テスト
        result = my_fl_pred.federated_eval(model, UPLOAD_DIR, mode=1)
        #response = {'accuracy': result}
        return render_template('acc_test.html', acc=result)
    else:
        response = {'message': "error"}
        return make_response(jsonify({'response': response}), 200)

    #return make_response(jsonify({'response': response}), 200)


if __name__ == "__main__":
    # app.run(debug=True)
    HOME_DIR = '/home/ubuntu/'
    app.run(debug=True, host='0.0.0.0', port=8888)
