# Ai心得

## [github修正原始碼](https://github.com/ian29285716/AI)

# 使用pytorch做數字辨識
* 我們專案中的其中一個分支本來想做紅綠燈的數字辨識，所以嘗試找了pytorch中有關數字的學習過程
* 參考了此網站 [toy demo - PyTorch + MNIST](https://xmfbit.github.io/2017/03/04/pytorch-mnist-example/)
### 使用pytorch內建的資料庫MNIST
  * 此資料庫是為手寫數字做辨識，隨然不符合原本專案目標，但辨識固定的LED數字應該會較容易，難處是要實現即時影線辨識
### 範例中建立了MLPNet和LeNet
  *  [MLPNet多層感知器](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-%E5%A4%9A%E5%B1%A4%E6%84%9F%E7%9F%A5%E6%A9%9F-multilayer-perceptron-mlp-%E5%90%AB%E8%A9%B3%E7%B4%B0%E6%8E%A8%E5%B0%8E-ee4f3d5d1b41)
  *  [LeNet捲積神經網路](http://noahsnail.com/2017/03/02/2017-3-2-LeNet%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/)
     * ![](https://i.imgur.com/maxyRex.png)
  *  兩者差異
    MLPNet將整張圖轉成了向量輸入，忽略一些圖像特徵
    Lenet 則是透過多層與算，提取出圖片的特徵
    [詳細資料](http://wiki.jikexueyuan.com/project/deep-learning/recognition-digit.html)
    
## 實作
* 原始碼都能從上方連結取得
* 取得數據
```python=
#使用顯卡
use_cuda = torch.cuda.is_available()

#對圖片做處理 
#ToTensor轉換成Tensor數據從灰階變成二元
#
trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (1.0,))])

#建立訓練資料，如果不存在電腦就下載
train_set = dset.MNIST(root=root,train=True,transform=trans,download=download)

##建立測試資料，如果不存在電腦就下載
test_set = dset.MNIST(root=root,train=False,transform=trans)

###訂定圖片尺寸
batch_size = 128

#DataLoader取得生成器
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

```

* 訓練模型
```python=
# 產生MLPNet
class MLPNet(nn.Module):
...略...
# 產生LeNet
class LeNet(nn.Module):
...略...


## training

#選擇模型
model = LeNet()

#使用顯卡
if use_cuda:
    model = model.cuda()
    
#優化器，可改變學習率
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
ceriation = nn.CrossEntropyLoss()

# 訓練過程
for epoch in range(10):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        ...略...
#  測試過程
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        ...略...

```

* 結果
  訓練批量 600
  測試批量 100
  
  損失函數 ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
  
  * LeNet
    各期測試結果

    epoch: 0, batch index: 100, test loss: 0.068039, acc: 0.000

    epoch: 1, batch index: 100, test loss: 0.041706, acc: 0.000

    epoch: 2, batch index: 100, test loss: 0.037343, acc: 0.000

    epoch: 3, batch index: 100, test loss: 0.032799, acc: 0.000

    epoch: 4, batch index: 100, test loss: 0.023330, acc: 0.000

    epoch: 5, batch index: 100, test loss: 0.021024, acc: 0.000

    epoch: 6, batch index: 100, test loss: 0.027192, acc: 0.000

    epoch: 7, batch index: 100, test loss: 0.021941, acc: 0.000

    epoch: 8, batch index: 100, test loss: 0.019057, acc: 0.000

    epoch: 9, batch index: 100, test loss: 0.021000, acc: 0.000
    
    [各期詳細訓練資料](https://hackmd.io/14TYEZiQT6i99Qi2vNEv-Q)
  * MLPNet
    各期測試結果

    epoch: 0, batch index: 100, test loss: 0.258626, acc: 0.000

    epoch: 1, batch index: 100, test loss: 0.177549, acc: 0.000

    epoch: 2, batch index: 100, test loss: 0.136973, acc: 0.000

    epoch: 3, batch index: 100, test loss: 0.110602, acc: 0.000

    epoch: 4, batch index: 100, test loss: 0.098599, acc: 0.000

    epoch: 5, batch index: 100, test loss: 0.097023, acc: 0.000

    epoch: 6, batch index: 100, test loss: 0.074558, acc: 0.000

    epoch: 7, batch index: 100, test loss: 0.074398, acc: 0.000

    epoch: 8, batch index: 100, test loss: 0.072555, acc: 0.000

    epoch: 9, batch index: 100, test loss: 0.073044, acc: 0.000

    [各期詳細訓練資料](https://hackmd.io/VyG7s0jHScWRncnnjL0nqQ)
    
   * 出現錯誤無法顯示出準確度
     因為pytorch版本問題須改編碼，但依[此網站](https://blog.csdn.net/weixin_41797117/article/details/80237179)嘗試改寫後仍無法顯示出準確度
    * 但單看測試集loss來看lenet較小但訓練時間較久，而MLPLNet則反之
     
# 使用OPencv
* 因為我本來的目標是想用opencv做出即時影像辨識，所以也採用了一些opencv的訓練與使用方式

## 使用opencv內建的XML檔做即時影像辨識與讀取影片
* 原始碼都能從上方連結取得
* 以人臉模型為例
* ![](https://i.imgur.com/ptskTZd.png)
* ![](https://i.imgur.com/CBYtqSn.png)

## 嘗試以自備資料訓練Haar模型
* Haar非常的簡易與輕便，早期的數位相機人臉辨識功能就是使用此種模型
* 參考
[如何使用-opencv_traincascade-訓練](http://tsuying.pixnet.net/blog/post/114235985-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8-opencv_traincascade-%E8%A8%93%E7%B7%B4)
[利用opencv训练基于Haar特征、LBP特征、Hog特征的分类器cascade.xml](https://blog.csdn.net/lql0716/article/details/72566839)

### 準備資料
![](https://i.imgur.com/3SzHoGI.png)

### 過程
* 因為opencv很多函式或程式都是寫成標準函式庫的方式，所以一開始使用上遇到極大的困難，後來學會直接使用cmd操作

### 結果
出現記憶體錯誤訊息或產生不出檔案，可能是樣本製作錯誤或軟體安裝位置問題
![](https://i.imgur.com/Xzv1lsT.png)
![](https://i.imgur.com/Dis1P57.png)
搜尋後發現有許多種可能，有可能是中文使用者名稱使過程出錯，這個可能要重新設定電腦才能解決

### 更新結果
因為負面樣本設定出錯，應該要設定的比正面樣本大。而且不限制圖片的大小與規定，經過設定後就能成功運行了
跑20層的HAAR的模型
![](https://i.imgur.com/uFNi1Uv.png)
實測抓紅燈可以抓到
![](https://i.imgur.com/pIuVbOK.jpg)


## 使用opencv內建圖像訓練KNN模型
* 因為使用第一個模型無法及時對數字做辨識，所以嘗試使用opencv內建的功能與自己手繪圖案來做訓練與測試
* 原始碼都能從上方連結取得
## 參考方法
[機器學習(1)--使用OPENCV KNN實作手寫辨識](http://arbu00.blogspot.com/2016/11/1-opencv-knn.html)

## 結果
以小畫家產生的圖片辨識率約為5成
![](https://i.imgur.com/7I0fqWi.png)
但可以持續擴大資料庫，就能改善個別的辨識效果


