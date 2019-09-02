本项目主要包括是 新闻人物言论提取 关键词提取 语义联想 词云四个模块，未来将继续迭代NER，文本摘要、情感分类等模块
本项目采取前后端分离开发，前端代码链接：https://github.com/BigErrors/NPSE  
                          后端采用Flask框架
![Image text](https://github.com/WenNicholas/nlp_assignment/blob/master/image/1.png)

一、新闻人物言论提取
.\my_project\speech_extraction\speechExtract.py                     
实施步骤：
1. 加载语料库
2. 加载模型（ltp分词、词性标注、依存句法分析）（这些在哈工大的ltp语言模型中都有的，只要安装好就可以用）
3. 根据上述模型和语料库（按行处理）得到依存句法关系parserlist
4. 加载预训练好的词向量模型word2vec.model
5. 通过word2vec.most_similar('说', topn=10) 得到一个以跟‘说’意思相近的词和相近的概率组成的元组，10个元组组成的列表
6. 仅仅是上面10个与‘说’意义相近的词是不够的，写个函数来获取更多相近的词。首先把第五步的‘词’取出来，把其后面的概率舍弃。取出来之后，按那10个词组成的列表利用word2vec模型分别找出与这10个词相近的词，这样广度优先搜索，那么他的深度就是10。这样就得到了一组以‘说’这个意思的词语组成的一个列表，绝对是大于10个的，比如说这些近义词可以是这些['眼中', '称', '说', '指出', '坦言', '明说', '写道', '看来', '地说', '所说', '透露',‘普遍认为', '告诉', '眼里', '直言', '强调', '文说', '说道', '武说', '表示', '提到', '正说', '介绍', '相信', '认为', '问', '报道']等。
7. 接下来可以手动加入一些新闻中可能出现的和‘说’意思相近的一些词，但是上面我们并没有找到的，比如‘报道’
8. 获取与‘说’意思相近的词之后，相当于已有谓语动词，接下来要找谓语前面的主语和后面的宾语了。由前面我们获取的句法依存关系，找出依存关系是主谓关系（SBV）的，并且SBV的谓语动词应该是前面获取的‘说’的近义词。那么接着应该找出动词的位置，主语和宾语的位置自然就找出来，就能表示了。那么怎么找位置？刚刚得到的依存关系是这样的[(4, 'SBV'),(4, 'ADV'),(1, 'POB'),(1, 'WP')]形式，前面的序号是取得主词的位置。主谓关系的主词是谓语，而且这个从1开始编号。所以我们就把符合上述要求的（主谓关系，并谓语动词是“说”的近义词）主语和谓语的id找出来。
9. 获得主语和谓语‘说’的序号之后，我们就要取得‘说的内容’也就是SBV的宾语。那么怎么寻找说的内容呢？首先我们看‘说’后面是否有双引号内容，如果有，取到它，是根据双引号的位置来取得。如果没有或者双引号的内容并不在第一个句子，那么‘说’这个词后面的句子就是‘说的内容’。然后检查第二个句子是否也是‘说的内容’，通过句子的相似性来判断，如果相似度大于某个阈值，我们就认为相似，也就认为这第二句话也是‘说的内容’。至此我们得到了宾语的内容。
![Image text](https://github.com/WenNicholas/nlp_assignment/blob/master/image/2.png)

二、关键词提取
.\my_project\keyword_extraction\keywordsExtract.py
采用TextRank方法提取文本关键词  
"""
       TextRank权重：

            1、将待抽取关键词的文本进行分词、去停用词、筛选词性
            2、以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
            3、计算图中节点的PageRank，注意是无向带权图
"""
![Image text](https://github.com/WenNicholas/nlp_assignment/blob/master/image/3.png)

三、语义联想
.\my_project\semanticSim.py
主要通过加载词向量，计算keywordsExtract.py得到的权重最大的两个词语，根据得到的词语计算近义词
其中词向量是基于维基百科训练得来，加上对新闻语料进行增量训练得到的
![Image text](https://github.com/WenNicholas/nlp_assignment/blob/master/image/4.png)

四、词云
.\my_project\word_statics.py
![Image text](https://github.com/WenNicholas/nlp_assignment/blob/master/image/5.png)



前端链接：https://github.com/BigErrors/NPSE




