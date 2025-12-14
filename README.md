

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

### Përshkrimi i Dataset-it

Pipeline-i tradicional u trajnua dhe u testua mbi dataset-in **HateXplain**, i cili përmban postime nga rrjetet sociale (kryesisht Twitter).

Karakteristikat kryesore të dataset-it:
- 3 klasa:
  - `normal`
  - `offensive`
  - `hateful`
- Rreth **16,000 mostra trajnimi**
- Rreth **4,000 mostra testimi/validimi**
- Çdo instancë ka **disa anotues njerëzorë**
- Etiketa finale përcaktohet me **majority voting**

### Sfida kryesore të dataset-it
- Subjektivitet i lartë i anotimit
- Mbivendosje semantike midis *offensive* dhe *hateful*
- Balancë jo e barabartë e klasave

Këto karakteristika e bëjnë HateXplain një dataset **sfidues**, veçanërisht për metodat tradicionale.

---

### I. Qasja Tradicionale e NLP-së (Traditional NLP Pipeline)

Për të përmbushur kërkesat e projektit dhe për të krijuar një bazë krahasimi me modelet neurale, u implementua një **pipeline i plotë tradicional NLP** për detektimin e gjuhës së urrejtjes. Kjo qasje përfaqëson metodologjinë klasike të përpunimit të tekstit, ku teksti transformohet në veçori numerike dhe më pas klasifikohet duke përdorur algoritme statistikore.

#### Pjesa e kodit
```text
.
├── src
    │   └── traditional_techniques
    │       ├── load_data.py
    │       ├── preprocess.py
    │       └── train_classical.py
```

Qasja tradicionale u përdor për të:
- Vendosur një **baseline të fortë**
- Analizuar kufizimet e metodave jo-neurale
- Krahasuar performancën me modelet neurale të mëvonshme


### Përpunimi i Tekstit (Text Preprocessing)

Përpunimi i tekstit është një hap kritik në pipeline-in tradicional, pasi cilësia e veçorive varet drejtpërdrejt nga pastrimi i të dhënave.

Hapat e ndjekur për përpunimin e tekstit:

#### 1. Normalizimi
- Kalimi i tekstit në shkronja të vogla
- Heqja e URL-ve
- Heqja e përmendjeve të përdoruesve (@username)
- Heqja e shenjave të pikësimit

#### 2. Tokenizimi
- Ndarja e tekstit në fjalë individuale (tokens)

#### 3. Heqja e Stopwords
- Eliminimi i fjalëve shumë të shpeshta dhe pak informuese (p.sh. *the*, *is*, *and*)

#### 4. Lemmatizimi
- Reduktimi i fjalëve në formën e tyre bazë (rrënjë)
- P.sh. *hating*, *hated* → *hate*


Shembull:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text
```    

Shembull i pastrimit te tekstit:
```
| Original Text                        | Cleaned Text        |
| ------------------------------------ | ------------------- |
| `@user This bitch in Whataburger!!!` | `bitch whataburger` |
```

**Arsyetimi:**
Qëllimi ishte të:
- Reduktohej zhurma
- Rritej përgjithësimi i modelit

---

### Krijimi i Veçorive (Feature Engineering)

Pas përpunimit të tekstit, fjalët u shndërruan në veçori numerike duke përdorur **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### TF-IDF Vectorization

U përdor:
- Unigramë (fjalë individuale)
- Bigramë (çifte fjalësh)

Kjo lejon kapjen e:
- Fjalëve individuale problematike
- Shprehjeve të shkurtra (p.sh. *hate you*, *go back*)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=30000
)

X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

```

**Pse TF-IDF?**
- Jep peshë më të lartë fjalëve diskriminuese
- Penalizon fjalët shumë të zakonshme
- Është standard në klasifikimin e tekstit

Megjithatë, TF-IDF:
- Nuk kap rendin e fjalëve përtej n-gramëve
- Nuk kupton kontekstin semantik
- Trajton çdo dokument si “bag-of-words”

---

## Modelet e Klasifikimit të Testuara

U testuan tre algoritme klasike të mësimit makinerik:

### 1. Logistic Regression
- Linear
- I shpejtë
- I interpretuar lehtë

**Vëzhgim:**
- Performancë stabile
- Kufizime në kapjen e marrëdhënieve komplekse

---

### 2. Support Vector Machine (Linear SVM)
- Maksimizon marginën ndarëse
- Efektiv për tekste të dimensioneve të larta

**Vëzhgim:**
- Performanca më e mirë ndër modelet tradicionale
- Megjithatë, ndjeshmëri ndaj klasave të mbivendosura

---

### 3. Multinomial Naive Bayes
- Bazuar në probabilitet
- Shumë i shpejtë

**Vëzhgim:**
- Performancë më e dobët
- Supozimi i pavarësisë së veçorive nuk vlen për gjuhën natyrore

---

## Konfigurimi i Eksperimentit

- Problem klasifikimi me **3 klasa**
- Ndarje train/test standarde
- Metrikat e përdorura:
  - Accuracy
  - **Macro-F1** (prioritet)

**Pse Macro-F1?**
- Dataset-i është i pabalancuar
- Macro-F1 penalizon performancën e dobët në klasat minoritare
- Jep pasqyrë më reale të cilësisë së modelit

---

## Rezultatet Eksperimentale

### Performanca e Përgjithshme

| Modeli | Accuracy | Macro-F1 |
|------|----------|----------|
| Naive Bayes | ~0.55 | ~0.52 |
| Logistic Regression | ~0.58 | ~0.56 |
| Linear SVM | ~0.60 | ~0.58 |

**Vëzhgime kryesore:**
- Linear SVM dha rezultatet më të mira
- Asnjë model tradicional nuk kaloi Macro-F1 ≈ 0.60
- Accuracy ishte metrikë mashtruese për shkak të klasës *normal*

---

### Analiza e Gabimeve

Modelet tradicionale performuan mirë në klasën:
- `normal`

Por patën vështirësi serioze në:
- Dallimin midis `offensive` dhe `hateful`

**Shembull i gabimit tipik:**
- ```“You people are disgusting”```
- Fjalia përmban fjalë fyese, por pa target grupor
- Modeli e etiketon gabimisht si *hateful*

Kjo ndodh sepse:
- TF-IDF nuk kap target-in semantik
- Nuk kuptohet konteksti ose qëllimi i fjalës

## Përfundim për Qasjen Tradicionale

Qasja tradicionale e NLP-së arriti rezultate të arsyeshme, por:
- Nuk mjafton për detektim të saktë të urrejtjes
- Dështon në raste kontekstuale
- Ka nevojë për modele më të thella dhe semantike

Kjo motivoi kalimin në qasje neurale, e cila përmirësoi performancën dhe kapacitetin interpretues të sistemit.



### II. Qasja neurale për klasifikim tre-klasor

Përveç pipeline-it tradicional të NLP-së, ky projekt implementon edhe një qasje të bazuar në rrjete nervore për zbulimin e gjuhës së urrejtjes. Qëllimi i këtij komponenti është të modelojë informacionet sekuenciale dhe kontekstuale në tekst, të cilat nuk mund të kapen në mënyrë efektive nga qasja tradicionale e cila përdor reprezentimet si bag-of-words ose n-gram.

Qasja nervore është implementuar duke përdorur PyTorch dhe përdor të njëjtin dataset (HateXplain) në një me klasifikim tre-klasor (normal, ofendues, gjuhë urrejtjeje).

#### Pjesa e kodit
```text
.
├── src
    │   └── three_class_lstm
    │       ├── ...
    │       .
    │       ....
```

**Përmbledhje e Modelit**

Modeli përfundimtar është një klasifikues dykahor LSTM i bazuar në mekanizmin e vëmendjes (attention-based bi-directional LSTM), me një shtresë të "embedding" me fjalë të trajnuara paraprakisht. 


Komponentet bazë:

- Tokenizimi dhe konstruktimi i fjalorit

- Pretrained GloVe word embeddings

- Enkoderi Bidirectional LSTM 

- Mekanizmi i "vëmendjes" për interpretim

**Përpunimi i të dhënave dhe Tokenizimi**

Përpunimi i tekstit për modelin nervor është qëllimisht minimal për të ruajtur sinjalet semantike:

- Shndërrimi i të gjitha shkronjave në të vogla

- Normalizimi i URL-ve (<URL>)

- Normalizimi i përmendjeve të përdoruesve (<USER>)

```python
def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub("<URL>", text)
    text = USER_PATTERN.sub("<USER>", text)
    return text.strip()
```

Tokenizimi kryhet duke përdorur një tokenizues të personalizuar që:

- Ndërton një fjalor nga të dhënat e trajnimit

- Zbaton një prag minimal të frekuencës

- Përdor tokenët <PAD> dhe <UNK>

- Konverton tekstin në sekuenca me gjatësi fikse përmes mbushjes (padding) ose shkurtimit (truncation)

Çdo hyrje (input) reprezentohet si një sekuencë e indekseve të fjalëve me një gjatësi maksimale prej 50 tokenësh.

**Shtresa 'Embedding'**
Fillimisht modeli ka përdorur embeddings të inicializuara në mënyrë të rastësishme, të mësuara nga dataseti i përzgjedhur. Kjo qasje ishte më e thjeshtë mirëpo më e ngadaltë dhe jepte performancë jo shumë të mirë. Përafërsisht: __Test Macro-F1 ≈ 0.50__.

Më pas, vendoset që modeli përdor embedding të trajnuara paraprakisht nga GloVe (200 dimensione) për të inicializuar shtresën e embedding-eve.

Ky ndryshim dha prodhoi një rritje në performancën e modelit në: __Test Macro-F1 ≈ 0.50-0.60__.

Arsyeja:

- Dataseti HateXplain nuk është mjaftueshëm i madh për të mësuar në mënyrë të besueshme reprezentimet semantike të fjalëve nga fillimi

- Shumë sinjale apo shenja që mund të tregojne urrejtjen zakonisht mbështeten në njohuri gjuhësore të përgjithshme (p.sh., ofendime, identifikues të grupeve të caktuara etnike/religjioze/etj.)

- Embedding-et e trajnuara paraprakisht përmirësojnë konvergencën dhe gjeneralizimin

Matrica e embedding-eve është e sinkronizuar me fjalorin e tokenizuesit, dhe embedding-et mund të “ngurtësohen” (përmes parametrave) gjatë trajnimit për stabilitet.

**Enkoderi LSTM**

Enkoderi kryesor është një LSTM dykahor, i cili përpunon tekstin në të dy kahjet, përpara dhe mbrapa. Kjo i mundeson modelit të kapë:

```python
self.lstm = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    bidirectional=True,
    batch_first=True
)
```

- Renditjen e fjalëve

- Varësitë afatgjate

- Kuptimin kontekstual brenda një fjalie

Konfigurimi:

- Hidden size: 128

- Numri i shtresave: 1

- Bidirectional: Po

- Formati i hyrjes/inputit: batch-first

Konfigurimi dykahor prodhon reprezentime të kontekstualizuara për çdo token në sekuencë.

**Mekanizmi i vemendjes**
Për të përmirësuar interpretueshmërinë dhe për të u fokusuar në sinjalet relevante, aplikohet një shtresë e 'vëmendjes' mbi output-et e LSTM-së.

```python
scores = self.attn(lstm_out).squeeze(-1)
weights = torch.softmax(scores, dim=1)
context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)

```

Mekanizmi i vëmendjes:

- Llogarit një vlerë/rezultat të rëndësisë (skalar) për çdo token

- Normalizon vlerat duke përdorur softmax

- Prodhon një reprezentim të peshuar të fjalive

Kjo i mundtëson modelit t'i dallojtë fjalët që janë më informuese për klasifikim, si fyerjet, ofendimet, ose referencat drejtuar grupeve të caktuara.

Kjo shtresë përmirëson modelin duke mundësuar të fokusohet në token-ët që shfaqin diskriminim dhe e redukton varësinë ndaj vetëm gjatësisë së sekuencës. Gjithashtu, mundëson interpretimin e rezultatit të parashikimit.

**Shtresa e Klasifikimit**

Reprezentimi i fjalive kalon neper:

- Dropout regularization

- Aktivizimit Softmax për parashikimin e 3 klasave

```python
self.dropout = nn.Dropout(0.3)
self.fc = nn.Linear(lstm_output_dim, num_classes)
```

Output-i përfundimtar korrespondon me klasat:

- 0 → normal

- 1 → ofendues

- 2 → gjuhë urrejtjeje


**Konfigurimi i Trajnimit**

Modeli trajnohet duke përdorur konfigurimin e mëposhtëm:

- Optimizer: Adam

- Funksioni i humbjes: CrossEntropyLoss

- Madhësia e batch-it: 32

- Shkalla e të mësuarit: 1e-3

- Epokat: deri në 8

Trajnimi përdor një ndarje të stratifikuar në setet e trajnimit, validimit dhe testimit për të ruajtur shpërndarjen e etiketave.

Modeli më i mirë zgjedhet dhe ruhet ne bazë të metrikës __Validation F1 Score__. 

**Rezultatet Eksperimentale dhe Përmirësimet**

**Modeli Neural Bazë**

- Pa embedding-e të trajnuara paraprakisht

- Pa mekanizëm vëmendjeje

- Test Macro-F1 ≈ 0.50

**Pas Shtimit të Embedding-eve GloVe**

Log-et e trajnimit treguan:

- Konvergjencë më e shpejtë

- Lakore te validimit më te qëndrueshme

Shembull:

```nginx
Epoka 4 | Train Acc 0.65 | Val Macro-F1 0.59

Epoka 5 | Train Acc 0.67 | Val Macro-F1 0.60
```


**Pas Shtimit të Mekanizmit të Vëmendjes**

Modeli arriti rezultatet më të mira:

Validation Macro-F1 ≈ __0.61__

Test Macro-F1 ≈ __0.60__

Test Accuracy ≈ __0.61__

Trajnimi përtej epokës 6–7 çoi në overfitting.

**Zgjedhja e Modelit Përfundimtar**

Modeli neural përfundimtar është një LSTM dypalësh me vëmendje dhe embedding-e GloVe të trajnuara paraprakisht, i zgjedhur sepse:

- Performon me mire se pipeline-i tradicional NLP

- Përmirëson modelin fillestar neural

- Ofron interpretueshmëri përmes mekanizmit të vëmendjes

**Kufizimet dhe te gjeturat**

- Ka konfuzion midis klasave ofenduese dhe gjuhë urrejtjeje


- Modele më të mëdha rrezikojnë overfitting në këtë dataset

Këto të gjetura motivojnë eksplorimin e ardhshëm të modeleve bazuar në transformer.


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

Rezultatet gjatë eksperimenteve tregojnë evoluimin e performancës së modeleve LSTM dhe CNN përgjatë epokave të trajnimit, duke përdorur metrikat Accuracy dhe Macro F1-score. Nga grafiqet Per-Epoch Performance vërehet se të dy modelet arrijnë konvergjencë relativisht të qëndrueshme, megjithatë CNN shfaq performancë pak më të lartë dhe më stabile në krahasim me LSTM, sidomos në Macro F1-score. LSTM ka luhatje më të theksuara përgjatë epokave, çka reflekton ndjeshmëri më të madhe ndaj ndryshimeve gjatë trajnimit.

Në krahasimin final të modeleve, CNN arrin Accuracy ≈ 0.705 dhe Macro F1 ≈ 0.697, ndërsa LSTM arrin Accuracy ≈ 0.691 dhe Macro F1 ≈ 0.682. Këto rezultate tregojnë se CNN ka avantazh të lehtë në këtë konfigurim eksperimental, veçanërisht për kapjen e pattern-eve lokale të gjuhës së urrejtjes. Megjithatë, të dy modelet demonstrojnë aftësi të krahasueshme për detektimin e gjuhës së urrejtjes, duke mbështetur përdorimin e qasjeve neurale si alternativa efektive ndaj metodave tradicionale NLP.