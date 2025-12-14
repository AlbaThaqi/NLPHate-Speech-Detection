

# Projekti : Detektimi dhe interpretimi i gjuhës së urrejtjes duke përdorur teknika tradicionale dhe neurale të NLP-së

**Lënda**: Procesimi i gjuhëve natyrale - NLP

**Profesori i lëndës**: Mërgim Hoti 

**Studimet**: Master - Semestri III

**Universititeti** : Universiteti i Prishtinës - " Hasan Prishtina "

**Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK

**Drejtimi** : Inxhinieri Kompjuterike dhe Softuerike -  IKS 


## Anëtarët e grupit:

- Alba Thaqi
  
- Fisnik Mustafa
  
- Rukije Morina

## Dokumentim Teknik i Projektit
Ky projekt ka për qëllim detektimin dhe interpretimin e gjuhës së urrejtjes (Hate Speech) duke përdorur dy qasje kryesore të NLP-së:

**Qasje tradicionale NLP** - bazohet në përfaqësimin e tekstit përmes veçorive statistikore dhe përdorimin e algoritmeve klasike të Machine Learning për klasifikim.

Algoritmet 

**TF-IDF** - është një metodë për përfaqësimin numerik të tekstit, e cila mat rëndësinë e një fjale në një dokument në raport me të gjithë koleksionin e dokumenteve.

**Algoritme klasike të Machine Learning** - Pas përfaqësimit të tekstit me TF-IDF, përdoren algoritme klasike të Machine Learning për klasifikimin e të dhënave.
Logistic Regression është një model linear që përdoret gjerësisht për klasifikim multiclass në NLP dhe Support Vector Machine ku është një algoritëm i fuqishëm për klasifikim që funksionon shumë mirë me data tekstuale të përfaqësuara me TF-IDF.

**Qasje neurale NLP** - Qasja neurale NLP bazohet në modele të rrjeteve nervore artificiale, të cilat mësojnë përfaqësime të thella të tekstit dhe kapin marrëdhënie kontekstuale midis fjalëve.

Algoritmet:

**Embedding** -Word embeddings janë përfaqësime dense dhe vektoriale të fjalëve, ku fjalët me kuptim të ngjashëm kanë vektorë të afërt në hapësirën vektoriale.
**CNN Model** - CNN përdoret si model alternativ neural për klasifikimin e tekstit. Ai aplikon filtra konvolucionalë mbi embeddings për të kapur pattern-e lokale (n-grams) që janë karakteristike për gjuhën e urrejtjes.CNN është veçanërisht efektiv për tekste të shkurtra dhe ofron trajnim më të shpejtë krahasuar me LSTM, por ka kufizime në kapjen e varësive kontekstuale afatgjata.
**Model LSTM** (Long Short-Term Memory)-LSTM është një variant i rrjeteve neurale rekurrente (RNN), i dizajnuar për të kapur varësi afatgjata në sekuenca.


Në këtë projekt, CNN përdoret për krahasim të drejtpërdrejtë me LSTM në aspektin e performancës dhe generalization.

 ## Struktura e projektit 
 data/ train.json / dev.json/ test.json
 
 preprocessing/text_preprocessing.py
 
 traditional_models/tfidf_vectorizer.py/ logistic_regression.py/ svm_model.py
 
 neural_models/ tokenizer.py/ lstm_model.py
 
 evaluation/metrics.py

## Dataseti 

Dataset-i i përdorur është HateXplain, i marrë GitHub
https://github.com/hate-alert/HateXplain

Ky dataset është krijuar posaçërisht për:

1.Hate speech detection

2.Offensive language detection

3.Explainability (interpretim)

### Pse kemi zgjedhur këtë dataset HateXplain?

Arsyet:
është dataset akademik, përmban annotime nga disa anotues dhe ofron fjalë të theksuara që shpjegojnë klasifikimin

Ky lloj dataseti lejon **detektim+interpretim** dhe jo vetëm klasifikim

### Përmbajtja e dataset-it
Dataset-i është në format JSON dhe përmban kolonat:

1. Fusha -	Përshkrimi

2. post_id	- ID unike

3. post_tokens -	Teksti i ndarë në fjalë

4. annotators	- Lista e anotuesve

5. label	Etiketa finale - (Hate / Offensive / Normal)

6. rationales	- Fjalët që justifikojnë etiketën

Dataset-i përmban rreth 40,000 shembuj.

**Dataset-i përmban tri klasa kryesore:**

1. Hate – përmbajtje që përmban gjuhë urrejtjeje

2. Offensive – gjuhë ofenduese, por jo domosdoshmërisht urrejtje

3. Normal – përmbajtje neutrale

**HateXplain ofrohet i ndarë paraprakisht në:**

- Train set – përdoret për trajnim të modeleve

- Development set (dev) – përdoret për validim dhe rregullim parametrash (kur aplikohet)

- Test set – përdoret vetëm për vlerësimin final të performancës

### Gjuhët programuese dhe libraritë e përdorura në kod 
I gjithë projekti është implementuar në gjuhën Python, duke përdorur librari të njohura për NLP, Machine Learning dhe Deep Learning.
Rezultatet e vlerësimit të modeleve ruhen në skedarë HTML, të cilët përmbajnë raporte të detajuara të metrikeve dhe vizualizime të performancës.

Libraritë e përdorura

1. pandas -	menaxhim i dataset-it (ngarkim JSON, manipulime)

2. numpy	- operacione numerike

3. scikit-learn	- TF-IDF vectorization, Logistic Regression, SVM, metrikat e vlerësimit

4. nltk -	stopwords, pastrimi i tekstit

5. torch – implementimi i modeleve neurale (CNN, LSTM), embedding dhe training

6. matplotlib / seaborn -	vizualizime (nëse përdoren)

7. json -	parsimi i skedarëve JSON të dataset-it HateXplain

8. re -	manipulim string-esh gjatë preprocessing


### Preprocessing i të dhënave
Qëllimi është pastrimi dhe normalizimi i tekstit.

Dataset-i HateXplain është në format JSON, ku secili rekord përmban token-at e postimit dhe anotimet.

Dataset-i HateXplain është i ndarë paraprakisht në train, development dhe test set.

1. Train set përdoret për trajnimin e modeleve

2. Dev set për rregullim të parametrave (opsional)

3. Test set përdoret vetëm për vlerësimin final

Në kod përdoret vetëm: post_tokens dhe label final

Rationales ruhen për analizë interpretimi, por nuk përdoren direkt për trajnim.

**Hapat që merren** teksti vjen i ndarë në token-a, prandaj hapi i parë është:
bashkimi i token-ave në një string të vetëm
Në kod kemi <img width="321" height="17" alt="image" src="https://github.com/user-attachments/assets/379cdf33-93fb-4323-a20e-bc27b4952b24" />

Roli i kësaj është që të bëhet tokenizer-i i LSTM-së të funksionojnë korrekt.

Në të njëjtin file (text_preprocessing.py) bëhen normalizimi që të mund të shmang dallimin midis psh: Hate dhe hate dhe njëkohësisht redukton dimensionin e veçorive

<img width="230" height="27" alt="image" src="https://github.com/user-attachments/assets/0d3c054f-b2f1-4e64-8b42-fa42f0187f44" />


Heqja e URL-ve dhe mentions -  URL-t dhe usernames nuk kontribuojnë në semantikën e urrejtjes por vetëm zvogëlojnë zhurmën në të dhëna

<img width="317" height="68" alt="image" src="https://github.com/user-attachments/assets/f2cb86a6-5b57-4cd4-b867-00b42be3e329" />

Heqja e karaktereve speciale - modelet tradicionale nuk përfitojnë nga simbole ruhet vetëm teksti semantik

<img width="337" height="38" alt="image" src="https://github.com/user-attachments/assets/a03472ab-6919-471f-9054-a284ca923f18" />


**Heqja e stopwords**
Përdoret lista standarde e NLTK stopwords ku fjalët si the, is, and nuk ndihmojnë në klasifikim
redukton dimensionin e vektorëve TF-IDF**

<img width="272" height="32" alt="image" src="https://github.com/user-attachments/assets/9d79fcf1-b9e5-4bc3-918b-26e8f3dbd396" />

(Opsionale) Stemming / Lemmatization i cili bashkon forma të ndryshme të së njëjtës fjalë përmirëson përgjithësimin e modelit

<img width="183" height="27" alt="image" src="https://github.com/user-attachments/assets/bc280867-11d2-47e4-a7ea-200fb9e10591" />

Pas preprocessing-ut, secili postim përfaqësohet si një string i pastruar dhe i normalizuar, i gatshëm për TF-IDF vectorization ose tokenization për LSTM.

Output i preprocessing-ut  kalon në dy pipeline të ndryshme: tradicional dhe neutral

I njëjti preprocessing përdoret si për qasjen tradicionale ashtu edhe për atë neurale, për të siguruar një krahasim të drejtë dhe të paanshëm mes modeleve.

Etiketat origjinale të dataset-it (Hate, Offensive, Normal) u mapuan në vlera numerike për t’u përdorur nga modelet e Machine Learning:

Hate → 0

Offensive → 1

Normal → 2

Nuk është aplikuar data augmentation apo balancing artificial i klasave, në mënyrë që modelet të vlerësohen mbi shpërndarjen reale të dataset-it.

## Pipeline Tradicional NLP
Qasja tradicionale NLP në këtë projekt bazohet në përfaqësimin statistikor të tekstit përmes TF-IDF dhe përdorimin e algoritmeve klasike të Machine Learning për klasifikim. 
Kjo pipeline shërben si baseline dhe ofron interpretueshmëri të lartë të rezultateve.

Pipeline tradicional përbëhet nga tre hapa kryesorë:

TF-IDF vectorization

Trajnimi i modeleve klasike të ML

Vlerësimi i performancës

### TF-IDF Vectorization

Shndërrimi i tekstit të pastruar në një përfaqësim numerik të përshtatshëm për algoritmet klasike të Machine Learning.

<img width="227" height="105" alt="image" src="https://github.com/user-attachments/assets/4f102476-fb44-4dd8-b064-56c298c00596" />

Teksti i pastruar merret si input nga preprocessing

Përdoret TfidfVectorizer nga scikit-learn ku krijohet një matricë sparse ku:
çdo rresht përfaqëson një dokument

çdo kolonë përfaqëson një fjalë ose n-gram

max_features = 5000
Kufizon madhësinë e fjalorit për të reduktuar kompleksitetin dhe overfitting.

ngram_range = (1, 2)
Përdoren unigram dhe bigram për të kapur shprehje të shkurtra karakteristike për gjuhën e urrejtjes.

### Ndarja Train / Test
Trajnimi i datasetit shmang overfitting dhe lejon vlerësim real të performancës.

<img width="377" height="26" alt="image" src="https://github.com/user-attachments/assets/b41bbd61-36b4-4637-9079-e184b2d45681" />

### Logistic Regression

Qëllimi
Klasifikimi i tekstit në klasat Hate, Offensive dhe Normal duke përdorur veçoritë TF-IDF.
TF-IDF vectors ndahen në train dhe test set

Trajnohet një model LogisticRegression

Gjenerohen parashikime për test set

max_iter = 1000
Siguron konvergjencë të modelit për data me dimension të lartë.

Modeli e mëson peshat e fjalëve ku çdo fjalë a ka ndikim pozitiv ose negativ në klasë
Si output ka accuracy, precision, recall, F1

<img width="341" height="27" alt="image" src="https://github.com/user-attachments/assets/88615800-5524-4bf3-86e6-9fc554424845" />


### Support Vector Machine (SVM)

Ofrimi i një modeli më të fuqishëm tradicional për klasifikim tekstual, për krahasim me Logistic Regression.
Kjo i merr të njëjtët TF-IDF vectors dhe përdor LinearSVC ku ndërton një hyperplane ndarës mes klasave

<img width="157" height="22" alt="image" src="https://github.com/user-attachments/assets/529ad7e3-c6a5-4ba1-9bca-8f8c859ec23d" />

Si output jep metrika vlerësimi për test set.


## Pipeline Neurale NLP
Qasja neurale NLP në këtë projekt përdor rrjete nervore artificiale për të mësuar përfaqësime të thella të tekstit dhe për të kapur kontekstin dhe rendin e fjalëve, çka është thelbësore për detektimin e gjuhës së urrejtjes në tekste reale.

Pipeline neurale përbëhet nga këto hapa:

Tokenization dhe Padding

Word Embeddings

Modeli LSTM

Trajnimi dhe parashikimi


#### Tokenization
Shndërrimi i tekstit të pastruar në sekuenca numerike që mund të përpunohen nga një model neural.
Përdoret Tokenizer nga Keras

Çdo fjalë mapohet në një ID numerik

Ruhet rendi i fjalëve në tekst

Sekuencat me gjatësi të ndryshme unifikohen përmes padding

<img width="318" height="73" alt="image" src="https://github.com/user-attachments/assets/1d5ff1cb-1dbb-4762-a50d-fcb582a85f70" />

Ndryshe nga TF-IDF, këtu rendi ka rëndësi.

#### Padding
Të sigurohet që të gjitha input-et kanë gjatësi fikse për modelin LSTM.
Padding aplikohet pas tokenization për të garantuar input uniform për modelin LSTM.

pad_sequences(sequences, maxlen=MAX_LEN)

<img width="400" height="43" alt="image" src="https://github.com/user-attachments/assets/e4934dd7-f9a3-43f4-ba2b-4e72c7912947" />


#### Embedding Layer

Shndërrimi i fjalëve nga ID numerike në vektorë dense që ruajnë informacion semantik ku çdo fjalë përfaqësohet si vektor numerik dhe embeddings mësohen gjatë trajnimit
Dimensioni i embedding-ut është fiksuar për të balancuar kompleksitetin dhe performancën.

<img width="378" height="36" alt="image" src="https://github.com/user-attachments/assets/d92da70a-7e90-44d8-b4fb-deb5548325f1" />

#### LSTM Layer
Kapja e varësive kontekstuale në tekst, lexon tekstin fjalë pas fjale dhe ruan informacion afatshkurtër dhe afatgjatë
LSTM(units=64)

<img width="201" height="33" alt="image" src="https://github.com/user-attachments/assets/e54d58b7-8e14-42c7-9c5a-da0c9f6ebe88" />

#### Dense + Softmax
Gjenerimi i parashikimit final të klasës.
probabilitet për secilën klasë:
Hate

Offensive

Normal

zgjidhet klasa me probabilitetin më të lartë
<img width="306" height="37" alt="image" src="https://github.com/user-attachments/assets/39fd90a5-693c-4f4e-a7c2-c9a6ae88a576" />

### Vlerësimi i Modeleve (Tradicional & Neural)

Metrikat e përdorura

Accuracy – saktësia e përgjithshme

Precision – sa parashikime pozitive janë të sakta

Recall – sa raste reale janë kapur

F1-score – balancë mes precision dhe recall

<img width="375" height="32" alt="image" src="https://github.com/user-attachments/assets/6a9d3ffd-c219-4bfc-839a-23b68c9e49d2" />


### Rezultatet 
Rezultatet tregojnë se LSTM arrin performancën më të lartë dhe më të qëndrueshme krahasuar me CNN dhe Transformer, veçanërisht në Macro F1-score, duke reflektuar kapjen më të mirë të kontekstit sekuencial. CNN performon mirë për pattern-e lokale, por has kufizime në varësitë afatgjata, ndërsa Transformer nuk ka arritur avantazh të qartë në këtë konfigurim eksperimental.
<img width="487" height="537" alt="image" src="https://github.com/user-attachments/assets/011526b0-cb16-415e-b6bb-a7f813e0f83c" />


<img width="497" height="736" alt="image" src="https://github.com/user-attachments/assets/3d9765c2-ca8a-4c4a-b6f3-bb29625a7e9a" />


<img width="732" height="483" alt="image" src="https://github.com/user-attachments/assets/aa049621-4877-437d-a1b7-48cbfccf2ee0" />


Modelet neurale ofrojnë performancë më të lartë dhe modelim më të avancuar të kontekstit krahasuar me metodat tradicionale NLP, ndërsa TF-IDF me Logistic Regression dhe SVM mbeten të rëndësishme për interpretueshmëri dhe baseline. Rezultatet mbështesin përdorimin e një qasjeje hibride, ku metodat tradicionale dhe neurale plotësojnë njëra-tjetrën.

<img width="746" height="637" alt="image" src="https://github.com/user-attachments/assets/7aec6872-39b7-4777-960f-74decfb4bd60" />

<img width="716" height="687" alt="image" src="https://github.com/user-attachments/assets/9c4859cc-bc93-4492-b077-e348b7556d07" />

<img width="737" height="767" alt="image" src="https://github.com/user-attachments/assets/bf64a2b3-1ac3-4174-ae11-fa3afa4a21ab" />

<img width="738" height="490" alt="image" src="https://github.com/user-attachments/assets/2d56da43-a8b4-4e00-b195-78e5b60855d8" />

