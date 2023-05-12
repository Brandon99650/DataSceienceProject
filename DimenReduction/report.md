# DS project2 報告

## 1. LDA & PCA:
PCA 是非監督式的降維方法，僅找尋投影后可以使資料分的最開的projection。
LDA 則式監督式學習，可以令組間variance最大，組內variance最小，對於分類則會比PCA更好

實驗結果

<img src=".\\PCA_LDA_cmp.png" />   

## 2. 心得
這次在做random projection的時候，我一開始很天真的想配置130107*10000的projection matrix，但當然沒有那麼大的RAM。所以我計算projection只好使用column by column的方式去實作，當然，跑很慢，但我目前還沒有查到有甚麼更好的方法，這也讓我體驗到效能跟space complexity之見間的trade off。

此外，在做LDA時，我一開始用自己隨機產生出來的「很小」的陣列
([ [1,2,3][4,5,6][7,8,9] ])
這樣去試，但馬上有問題: $S_w$ 是 singlar，我上網查後發現是說要對他進行normalize 之類的，但由於iris dataset 沒有這個問題，所以我沒有額外處理。這也讓我知道在使用LDA時，資料要越齊全越好。