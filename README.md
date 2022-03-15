# OCR 光学字符识别
本项目使用CTPN模型进行文本检测CRNN模型进行文本识别，实现了基于pytorch的OCR项目。
用户可以上传一张带文字的图片，我们将返回图片的文字检测效果图片和文本形式的内容。
## 1. 引入
光学字符识别（Optical Character Recognition, OCR）是指对文本材料的图像文件进行分析识别处理，以获取文字和版本信息的过程。也就是说将图象中的文字进行识别，并返回文本形式的内容。  
例如（该预测效果基于OneFlow云平台OneCloud一键OCR识别效果展示）：
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/156345011-d0d9f305-f0d0-45db-8930-1b9633c87dd8.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/156346390-92957986-18f1-4966-b268-714b609518cf.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

### 1.1. OCR应用场景
根据OCR的应用场景而言，我们可以大致分成识别特定场景下的专用OCR以及识别多种场景下的通用OCR。  就前者而言，证件识别以及车牌识别就是专用OCR的典型案例。  针对特定场景进行设计、优化以达到最好的特定场景下的效果展示。 那通用的OCR就是使用在更多、更复杂的场景下，拥有比较好的泛性。在这个过程中由于场景的不确定性，比如：图片背景极其丰富、亮度不均衡、光照不均衡、残缺遮挡、文字扭曲、字体多样等等问题，会带来极大的挑战。 现OneCloud为大家提供的是通用OCR模型，支持中英文数字组合识别、长文本识别场景
### 1.2. OCR技术路线

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/156347566-6767c09d-6337-425b-b4ca-2b22e3ae18e1.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

其中OCR识别的关键路径在于文字检测和文本识别部分，这也是深度学习技术可以充分发挥功效的地方。本项目的预训练模型的网络结构是CTPN+CRNN
CTPN是在ECCV 2016提出的一种文字检测算法。CTPN结合CNN与LSTM深度网络，能有效的检测出复杂场景的横向分布的文字，是目前比较好的文字检测算法。
下图是CTPN的网络结构模型

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/156347367-cf8087df-3d43-46a7-b997-1c6365ba5b8b.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

接着，我们使用 CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络，是DCNN和RNN的组合，专门用于识别图像中的序列式对象
下面是CRNN的网络结构模型

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/156347184-8625cc56-00b2-4abb-8db9-503db5add643.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

## 2. 项目文件说明

``` text
├───model                   # 预训练模型
├───templates               # 前端页面
├───detect
│   │   config.py           # 配置类
│   │   ctpn_predict.py     # 预测类
│   │   ctpn_model.py       # 模型源码
│   └───ctpn_util.py        # 工具类
├───recognize
│   │   config.py           # 配置类
│   │   crnn_recognizer.py  # 预测类
│   │   crnn.py             # 模型源码
│   └───keys.py             # 工具类
├───requirements.txt        # Python 依赖包
├───ocrinfer                # 推理
└───server.py               # 后端启动程序
```

## 3. WEB系统架构

项目使用 Flask 来搭建 WEB 应用，连通用户前端，后端服务，和算法推理。

`server.py` 采用 `Flask` 在服务器端启动过一个 Web 服务。这个 FLASK 服务器前接用户来自浏览器的请求，后接用于推理图片结果的OCR对象 。整体架构如下所示：

```text
┌───────┐           ┌───────┐        ┌─────────┐
│       │    AJAX   │       │        │         │
│       ├───────────►       ├────────►         │
│ USER  │           │ FLASK │        │   OCR   │ 
│       ◄───────────┤       ◄────────┤         │
│       │    JSON   │       │        │         │
└───────┘           └───────┘        └─────────┘
```

### 3.1. 前端

前端实现代码在```templates/home.html```中，它提供了按钮```<上传图片>```用于用户拍照/上传图片：

```
<van-uploader id="image" name="image" :after-read="afterRead" :max-count="1" />
```

当用户拍照/上传图片后，通过axios发送 POST 请求给后端，并且将后端返回的预测结果更新到前端页面上：
```
axios.post("", formData).then((res) => {
                console.log('Upload success');
                vue_obj.image_url = res.data.image_url;
                vue_obj.prediction = res.data.prediction;
                vant.Toast(res.data.prediction);
        });
```
因为用户拍摄的照片往往较大，所以上传照片前，会预先对照片进行压缩：
```python
new Compressor(file.file, {
              //...
            });
```

### 3.2. 后端

在 server.py 中，注册了路由```/v1/index```：

```python
@app.route(f"{base_dir}/v1/index", methods=["GET"])
#...
```

当接受 GET 请求时，返回主页面（home.html）。当接受 POST 请求时，则做两件事情：

- 保存图片（方便之后被前端引用显示）；
- 调用模型对图片进行预测并返回预测结果；

保存图片相关代码：

```python
filename = generate_filenames(image_file.filename)
filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
image_file.save(filePath)
```

对图片进行预测相关代码：

```python
def predict(filename):
    print('filename:'+filename)
    original_image_url = url_for("images", filename=filename)
    original_image_file_path = os.path.join(
        app.config["UPLOAD_FOLDER"], filename)
    result=[]
    try:
        out, output_file = model.ocr(original_image_file_path)
        for key in out:
            result.append(out[key][1])
        json_obj = {"image_url": original_image_url, "prediction": result}
        return json.dumps(json_obj)
    except:
        json_obj = {"image_url": original_image_url, "prediction": original_image_file_path}
        return json.dumps(json_obj)

```

### 3.3. 推理

后端推理使用PyTorch，通过读取已经训练好的模型文件```model/CTPN.pth```初始化CTPN模型、```model/CRNN.pth```初始化CRNN模型。

```python
detect/ctpn_predict.py
weights = os.path.join(config.checkpoints_dir, 'CTPN.pth')
model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
```
```
recognize/crnn_recognizer.py

```

输入待识别的图片，模型进行推理并输出识别结果：

```python
@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def home():
        ...
        if image_file and is_allowed_file(image_file.filename):
            try:
                filename = generate_filenames(image_file.filename)
                filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(filePath)
                return predict(filename)
            except Exception:
                json_obj = {"image_url": "", "prediction":"后台异常"}
                return json.dumps(json_obj)
```
  
                

## 5. 项目部署和使用
#### step1:点击项目中的部署
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157579048-ae547d83-2fa1-43d2-a266-aadbcad3ad52.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

#### step2:选择模型文件，选中"model"文件 点击下一步

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157579197-6b0c2b88-0bbf-4bcf-afa0-d99f5f25f585.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

#### step3:填写基本信息

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157579401-9711536e-74e3-4206-a827-331c561b80d9.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

#### step4:填写配置信息

     部署环境选择公开环境的'oneflow-0.6.0+torch-1.8.1-cu11.2-cudnn8'
     启动命令行填写 'cd /workspace && ./run.sh'
     部署成功后选择 ‘运行’
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157579468-9185bb9f-5cf2-4dbc-a81a-c67f054c40ba.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157579868-e1254b1f-641b-40dc-bdf0-ae684a5fcb6d.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

#### step5:选择运行环境

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157579893-b2287384-b0aa-4d06-96a3-2cb31022c326.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

等待项目启动成功

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157580293-4a8cf14e-a83b-4562-92c2-4948b103a6a6.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

#### step6:使用

点击测试
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157580489-ee791022-dfa8-4372-9645-b1653d955874.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>
等待2分钟后 点击上传你所要识别的图片
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157580602-f8e6c3b8-924d-41df-a0f8-626f3012dfff.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>
将会输出识别和检测结果
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/48576019/157581062-a3f28c87-4145-48a3-83b2-0fc9b847f5d0.png">
    <br>
    <div style="color:orange; 
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

## 6.在服务器端进行推理
在项目model目录下执行
```python
python3 ocrinfer.py image<需要识别的图片路径> result_dir<结果保存路径>
```







