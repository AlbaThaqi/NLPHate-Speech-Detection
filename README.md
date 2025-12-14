

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

**Dataset-i përmban tri klasa kryesore:**

1. Hate – përmbajtje që përmban gjuhë urrejtjeje

2. Offensive – gjuhë ofenduese, por jo domosdoshmërisht urrejtje

3. Normal – përmbajtje neutrale

### Përmbajtja e dataset-it
Dataset-i është në format JSON dhe përmban kolonat:

1. Fusha -	Përshkrimi

2. post_id	- ID unike

3. post_tokens -	Teksti i ndarë në fjalë

4. annotators	- Lista e anotuesve

5. label	Etiketa finale - (Hate / Offensive / Normal)

6. rationales	- Fjalët që justifikojnë etiketën

Dataset-i përmban rreth 40,000 shembuj.

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



## III. Qasja alternative neurale (Klasifikim baze binar)

### Formulimi i problemit: Binary Classification

Ndryshe prej modelit të paraqitur në II, i cili performon **three-class classification** (`normal`, `offensive`, `hateful`), ky implementim alternativ e riformulon problemin si **binary classification**, për të parë dallimet në performancë.

### Mapimi i etiketave

| Original Label | Binary Label |
|---------------|-------------|
| normal | 0 (non-hateful) |
| none | 0 (non-hateful) |
| offensive | 1 (hateful/toxic) |
| hatespeech | 1 (hateful/toxic) |

Mapimi implementohet direkt në kodin për ngarkim të të dhënave

```python
y = 0 if example["label"] in ("normal", "none") else 1
```

Si rezultat modeli mëson të dallojë përmbajtjet **toxic vs non-toxic**
---

### Arkitektura e implementuar

Ky implementim alternativ vlereson disa arkitektura neurale duke shfrytëzuar të njëjtin piepline të trajnimit dhe vlerësimit
- LSTM-based classifier  
- CNN-based classifier  
- Transformer-based classifier

Të gjithë modelet implementohen përmes PyTorch

---

### Arkitektura LSTM

Ky model LSTM dallon nga ai i prezantuar në pjesën II.

#### Architecture Details

- Bidirectional LSTM
- Max pooling over time
- Randomly initialized embeddings
- Binary output layer

```python
self.lstm = nn.LSTM(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    bidirectional=True,
    batch_first=True
)

self.fc = nn.Linear(hidden_dim * 2, 2)
```

#### Dallimet kryesore nga PII

| Aspect | Part II LSTM | Part III LSTM |
|------|------------|---------------|
| Task | 3-class | Binary |
| Embeddings | GloVe (pretrained) | Random |
| Attention | Yes | No |
| Pooling | Attention | Max pooling |
| Output Classes | 3 | 2 |

---

### Arkitektura CNN

Modeli baze CNN perdor konvolucioned 1D per te kapur paternat lokale n-grame:

```python
self.conv = nn.Conv1d(
    in_channels=embed_dim,
    out_channels=128,
    kernel_size=3
)
```

CNN fokusohet ne paternat leksikore dhe sinjalet e toksicitetit te bazuara ne fjale kyce.

---

### Konfigurimi i trajnimit

- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Batch size: 32  
- Learning rate: 1e-3  
- Epochs: up to 8  
- Metric: Macro F1  

---

### Rezultatet e eksperimentimit

#### LSTM Results (Binary)

Performanca me e mire u arrit ne epokat 3–4:

- Accuracy ≈ **0.73**
- Macro F1 ≈ **0.72**

Epokat tjera treguan overfitting.

### CNN Results (Binary)

- Accuracy ≈ **0.71**
- Macro F1 ≈ **0.70**

CNN tregoi performance me stabile edhe pse pak me te ulet se LSTM.

---

## Krahasimi me LSTM nga P.II

| Model | Task | Macro F1 | Accuracy |
|------|------|----------|----------|
| Main LSTM (Part II) | 3-class | ~0.60 | ~0.61 |
| Binary LSTM (Part III) | Binary | ~0.72 | ~0.73 |
| Binary CNN (Part III) | Binary | ~0.70 | ~0.71 |

Rezultatet me te mira ne P.III vijne pershkak te klasifikimit binar krahasuar me ate tre-klasor.

---

### Permbledhje 

Kjo qasje alternative neurale demostron se:

- BKlasifikimi binar rezulton me metrika me te mira mirepo me me pak nuance  
- Modelet baze neurale performojne me mire se modelet tradicionale te NLP  
- Formulimi i problemit ndikon ne performance 



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