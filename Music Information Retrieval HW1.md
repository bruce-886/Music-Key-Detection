# **Music Information Retrieval HW1**
<!-- ## 學號 : 108033592 姓名 : 張庭宇 系所 : 清大動機所 -->

<!-- --- -->

### **Question 1.**
#### GTZAN dataset:

    Total accuracy : 26.52%
    blues   :  7.14%
    country : 32.32%
    disco   : 31.63%
    hiphop  : 13.58%
    jazz    : 16.46%
    metal   : 24.73%
    pop     : 41.49%
    reggae  : 32.99%
    rock    : 34.69%
    
在blues跟hippop分類下利用binary template matching 效果不如其他分類的好，原因有可能是因為整體的音調偏低，同時又有加入鼓聲或是bass等低音樂器造成低音聲部能量過強導致誤判。
#### Giantsteps dataset:
    Total accuracy : 28.05%
    
![](https://i.imgur.com/TgphWHH.png)

圖片來源 https://www.kaggle.com/imsparsh/gtzan-genre-classification-deep-learning-val-92-4#Data-Visualization

由各特徵之間相關係數圖可以得知mfcc_2的平均值對chromagram的平均值是有負相關的情形。由於mfcc_2是低頻強度的總和係數，因此對低頻強度高的聲音訊號，可能會有chromagram特徵不明顯的情況，導致誤判的情況。



### **Question 2.**
![](https://i.imgur.com/Lpan7qQ.png)
![](https://i.imgur.com/0HZCe9O.png)

由於人類的聽覺系統並非線性的，因此加入對數轉換後可以比較符合聽覺的感知大小。但在此並沒有觀察到準確率進步的現象，原因推測為聽覺的感知不單單只受到音高影響，也會根據樂器的不同或是唱腔的不同而改變，因此直接針對對數刻度下的chromagram 做預測不一定能獲得準確率的提升。同時從數學的角度來看，加入對數刻度後反而壓制了特徵數值大的 chroma vector ，造成誤判的機率隨之提升。


### **Question 3.**
#### GTZAN dataset:
    Weighted accuracy : 41.15%
    blues   : 19.59%
    country : 54.85%
    disco   : 49.49%
    hiphop  : 20.25%
    jazz    : 31.90%
    metal   : 35.38%
    pop     : 56.60%
    reggae  : 47.84%
    rock    : 49.29%
#### Giantsteps dataset:
    Weighted accuracy : 41.12%

在進行調性預測時，常常因為高音諧波導致誤判，因此根據音樂調性理論建構加權評分機制以達到更好的評價標準。如下面的表格以及各調性分數對應表。
![](https://i.imgur.com/lDelKoR.png)

![](https://i.imgur.com/jEqy7JC.png)



### **Question 4.**
#### GTZAN dataset:
    Total accuracy : 38.83%
    blues   : 19.39%
    country : 49.49%
    disco   : 34.69%
    hiphop  : 18.52%
    jazz    : 29.11%
    metal   : 34.41%
    pop     : 58.51%
    reggae  : 57.73%
    rock    : 42.86%
    --------------------------
    Weighted accuracy : 51.94%
    blues   : 30.00%
    country : 68.08%
    disco   : 52.76%
    hiphop  : 29.38%
    jazz    : 41.90%
    metal   : 47.31%
    pop     : 67.87%
    reggae  : 66.80%
    rock    : 57.86%
![](https://i.imgur.com/XjTdIjJ.png)

#### Giantsteps dataset:
    Total accuracy : 42.74%
    --------------------------
    Weighted accuracy : 53.52%
![](https://i.imgur.com/jKu0wxu.png)
- Binary template matching v.s Krumhansl-Schmuckler’s method
    Krumhansl-Schmuckler’s method 是利用人類聽覺數據統計出來的調性相對參數值，因此相對於binary template matching ，Krumhansl-Schmuckler’s method 提供了更為貼近人類聽覺的調性比對模板。故在判斷準確率上有著明顯的進步，因為其統計數據考慮到了人類感知系統相較於現實有著非線性的差異。
- Any limitations of these two methods?
    利用binary template matching在決定主音時是利用chroma vector裡最大值的位置當作主音。但此做法可能受到樂器種類或是音樂本身性質影響。而Krumhansl-Schmuckler’s method 則是針對所有的調性做相關性分析，找到最大相關係數值作為其預測之調性，此方法準確度較高，但因為要計算所有模板的相關係數，故計算量比較大。
- Any limitations of using GTZAN dataset for key finding?
    Sturm在2014年發表的文獻裡提到GTZAN dataset有三項問題: 
    1. 重複資料
    2. 曲風分類標註錯誤
    3. 聲音訊號扭曲

    在本次的調性判斷任務中，主要會影響到結果的為第一項以及第三項。首先第一項錯誤會使得我們對於調性判斷的準確度產生誤解，譬如某項曲風的調性判斷準確度非常高有可能只是因為重複採樣而導致誤解。再來第三項錯誤可能會導致計算chromagram 時得到錯誤的頻率值，進而推測出錯誤的主音及調性。另外在GTZAN dataset中各音訊都只有30秒，對於計算出穩定的chroma-gram是有一定難度的。
    - 參考論文 : Sturm, B.L., 2014. The State of the Art Ten Years After a State of the Art: Future Research in Music Information Retrieval. Journal of New Music Research 43, 147–172.. doi:10.1080/09298215.2014.894533
    
### **Question 5.**
#### Methodology:
為了檢測 local key，在此利用滑動窗口的概念對當下的chromagram做擷取，其概念圖如下圖。

![](https://i.imgur.com/WGxCZt9.png =600x500)

其中kernel權重可以有不同分布，如常數分布("constant")、線性分布("linear")或是指數分布("log")。
對於超出原本長度的部分做了padding，邊界的padding設置了兩種模式，分別為補零("constant")與延續邊界("edge")。

![](https://i.imgur.com/aBjBdcK.png =300x200)![](https://i.imgur.com/dC1G5f4.png =300x200)

![](https://i.imgur.com/gvOTj4h.png =300x200)

接著對於經過權重換算過後的特徵分別進行binary template matching 和 Krumhansl-Schmuckler’s method計算出預測結果。


---
#### BPS-FH dataset:
在此題中，將不同的segment size對準確率作圖。
* Weighting = "constant", Padding = "constant"
![](https://i.imgur.com/F89YtQs.png =600x300)

* Weighting = "linear", Padding = "constant"
![](https://i.imgur.com/9KMUqY3.png =600x300)

* Weighting = "linear", Padding = "edge"
![](https://i.imgur.com/icWOlkm.png =600x300)

* Weighting = "log", Padding = "edge"
![](https://i.imgur.com/dBCCZvj.png =600x300)

可以觀察到大約在segment size=35附近可以得到不錯的效果，同時計算量也維持在較低的範圍。

經過數據觀察後，可以發現對於kernel的權重為線性分布以及指數分布得到的結果差不多，因為其權重的尺度比例差不多。同時觀察設置padding為延續邊界可以增加端點的預測準確率，推測是其提升了邊界特徵的明顯度。

* BPS-FH dataset local key detection result
```
    Binary accuracy : 23.34%
    --------------------------
    Binary weighted accuracy : 40.85%
    --------------------------
    KS accuracy : 41.97%
    --------------------------
    KS weighted accuracy : 54.55%
```
![](https://i.imgur.com/gMqWzOM.png)
![](https://i.imgur.com/5foo3cD.png)

從混淆矩陣中可以看到其圖案接近於加權的分數表，而相對於利用binary template matching，使用Krumhansl-Schmuckler’s method的混淆矩陣對角線上的值更大更靠近正確的預測，而且整體集中在幾個區域，達到比較穩定的預測。



#### A_MAPS dataset:
和BPS-FH dataset做法一樣。對於不同參數下作圖、觀察後得到結果如下。

* Weighting = "constant", Padding = "constant"
![](https://i.imgur.com/HDAlgOu.png)

* Weighting = "linear", Padding = "edge"
![](https://i.imgur.com/5H9212k.png)


* A_MAPS dataset local key detection result
```
    Binary accuracy : 19.02%
    --------------------------
    Binary weighted accuracy : 32.81%
    --------------------------
    KS accuracy : 29.24%
    --------------------------
    KS weighted accuracy : 43.18%
```
觀察到大約在segment size=40附近可以得到不錯的效果

### **Question 6.**
#### 利用GTZAN dataset搭配 CNN 做tonic detection
Methodology :
將chromagram上每30個frame的data加總在一起，形成一個12x40的矩陣，利用卷積神經網路做訓練，模型架構如圖。

<!-- ![](https://i.imgur.com/cjntgpd.png) -->
![](https://i.imgur.com/riItfBQ.png)

![](https://i.imgur.com/qQccVyL.png)
```
    Test acc ~ 40%
    --------------------------
    Weighted acc ~ 45%
```

![](https://i.imgur.com/0INrjRV.png)


#### 利用BPS-FH dataset搭配 xgboost 做local key detection
Methodology :
將每一秒的前後各10秒的chromagram加總後當作輸入，並利用xgboost做local key detection

```
    Test acc : 40.52%
    --------------------------
    Weighted acc : 49.05%
```

![](https://i.imgur.com/V6YSLVl.png)

![](https://i.imgur.com/XAH47JK.png)

---
#### 討論:
從結果上來看，上述的兩種做法大概都跟利用Krumhansl-Schmuckler’s method差不多準確度，但是若是比較經過加權的分數來說，利用學習的方法很難利用好加權的機制提升準確率，因為模型訓練的方式是只以答對答錯來區別。
對於上述的兩個問題，雖然都是以非常簡單的模型做預測，也不涉及過多的特徵工程，仍然獲得了可接受的準確度。以下是幾個我認為還能改進的點:
1. 大部分的dataset都是imbalanced dataset應該可以做label smoothing或是over/undersampling來減緩這個問題
2. 應該嘗試加入KS的參數進行預測，用以更好的利用已知的特徵擷取器
3. 應該要加入data augmentation，增強模型的泛化程度
4. 加入更多的人為設定的特徵參數，如mfcc等，用以增加判斷的依據
5. 目標函數可以考慮用加權的方式，而非只有對錯






    
    
    
    
    
    
    
