# Prompts

LETITBE has two prompts, and both are part of its specification rather than implementation
detail. This file reproduces them verbatim so that the metric can be read, cited and checked
without reading Python. The `selfcheck` subcommand of each script asserts that the text here
still matches the constant it ships, so the two cannot drift apart.

## Why these are specification

The paper states that the engine is part of the specification. The prompt is the other half of
that claim: a reworded prompt, or one with its fields in a different order, is a different
metric whose numbers do not compare with the published ones.

Translating these prompts is therefore not a localisation task. The source text is Taiwanese
legal Chinese, and the aspect names are quoted from TIPO's examination guidelines.

## 1. Hit detection, used at scoring time

Asked once per gold reason point. The engine sees one candidate explanation and one reason
point, and answers whether the candidate asserts it. Defined in `letitbe.py` as `_RULE` and
`_FIELDS`, and assembled by `qa_text`.

The expected answer is a single character, 是 or 否. A longer reply is read by its first 是 or
否, with 不是 read as 否, and a reply with neither is read as 否: an engine that hedges is an
engine that did not assert the point.

### Rule

```
你是商標近似論述的命中判定器。判斷「候選解釋」有沒有做出指定的「理由論斷」，輸出 是/否。
標籤：是 = 候選有做出該論斷（語意蘊含即可，用字不必相同）；否 = 沒提到、或講相反／矛盾。

該論斷屬於下列 aspect 之一（用語依商標近似審查基準 §5.2，括號為基準原文範例）：
- 主要識別部分：消費者關注、事後留印象中較顯著的部分；含主要字詞與形容字詞之分。指名兩商標共同的字／外文／圖形。（例：「泰山」與「小泰山」主要字詞均為「泰山」）
- 差異：些微／細微差異；不可機械式比對。指名差在哪。（例：「house」與「horse」可能近似，「house」與「mouse」則否）
- 外觀：文字商標就外觀觀察是否相像（常同時斷言「在外觀」）。
- 近似程度結論：構成近似與否及其近似程度（高／不低／低）。
※ 主要識別部分與差異常同句出現，例：「『功夫』二字外觀近似」→ 主要識別部分「共同含『功夫』」＋外觀「外觀近似」。
範例命中判定：論斷「兩商標共同含『鍋寶』」、解釋「二者都有鍋寶二字」→ 是；論斷「外觀近似」、解釋「外觀並不近似」→ 否。
```

### Fields appended to the rule

```
【aspect】{aspect}
【理由論斷】{claim}
【候選解釋】{cand}
```

The three placeholders are filled per item, in this order, and the order is part of the
specification.

## 2. Decomposition, used at build time

Run once, offline, when a corpus is built, at temperature 0 with the aspect field pinned to an
enum by a response schema so the model cannot invent a label. Its output is the frozen gold
checklist. Defined in `preprocess.py` as `PROMPT`.

The decomposition is part of the dataset definition rather than a variable, so it is not
ablated. Re-running it with a different model, or at a different temperature, produces
a different dataset, and the audit figures reported for the published checklists do not carry
over to it.

```
你是商標近似論述的結構化標註員。輸入一段 TIPO 核駁處分書的「商標近似」論述，拆成「審查官判斷此案為何近似的理由點」清單。
【不要抽（每案都有的套語，不帶此案資訊）】
- 抽象定義開場：「商標近似係指…整體印象有相近之處」
- 觀察方法：以普通知識經驗消費者、施以普通注意、異時異地隔離觀察、連貫唱呼
- 制式混淆結尾：「可能誤認…來自同一來源或雖不相同但有關聯之來源」← 每案都有的套語、是混淆「結果」非近似「理由」，跳過不抽
（但：該案構成近似/近似程度高之判斷要抽，見下。）
【aspect 只能選下列 4 個之一】（用語依審查基準 §5.2，括號為基準原文範例）
A. 指名「哪個字/部分」，回答「兩商標共同的是哪個元素」
   - 主要識別部分：「主要部分」＝消費者關注、事後留印象中較顯著的部分；含主要字詞與形容字詞之分。抽：指名共同的字/外文/圖形。（例：「泰山」與「小泰山」主要字詞均為「泰山」）
	商標混淆誤認審查基準：判斷商標近似，應以商標圖樣整體為觀察。此乃係由於商標呈現在商品/服務之消費者眼前的是整體圖樣，而非割裂為各部分後
分別呈現。所謂「主要部分」觀察，則係商標雖然以整體圖樣呈現，然而商品/服務之消費者關注或者事後留在其印象中的，可能是其中較為顯著的部分，此一顯著的部分即屬主要部分。
B. 指名「哪個字/部分」，回答「兩商標相異的是哪個元素」
   - 差異：些微差異/細微差異；不可機械式比對。抽：指名差在哪。（例：「house」與「horse」可能近似，「house」與「mouse」則否）
	商標混淆誤認審查基準：判斷近似另一重要原則為異時異地、隔離觀察原則。所謂一般實際購買行為態樣，係指一般消費者都是憑著對商標未必清晰完整的印象，在不同的時間或地點，來作重覆選購的行為，而不是拿著商標以併列比對的方式來選購，所以細微部分的差異，在消費者的印象中難以發揮區辨的功能，判斷商標是否近似時，得無庸納入考量。這樣一個設身處地的思考原則，在判斷商標近似時，應注意切實掌握。
C.  判斷「在外觀面向是否相像」：
   - 外觀：文字商標就外觀觀察。常同時斷言（「在外觀」）。
商標混淆誤認審查基準：傳統商標的構成要素，包括文字、圖形、記號或其聯合式等，其書寫方式、組合態樣、構圖設色、相對位置、比例大小等情形，均可能影響消費者對於該商標的注意程度，判斷商標是否構成近似，原則上以商標註冊申請之商標圖樣為準據，就商標整體外觀方面，判斷是否已達到可能誤認的近似程度，而非單純機械性比對。
D. 綜合關係
   - 近似程度結論：構成近似與否及其近似程度（高/不低/低）。
	商標混淆誤認審查基準：商標近似程度的認定，必須考量構成商標要素的識別性，及商標予消費者之整體商業印象加以判斷。商標在外觀、觀念或讀音等方面，若存在相似之處越多，近似程度就越高，反之越低。商標雖同為構近似，然僅讀音近似者，或讀音、外觀及觀念均近似；其近似之程度即有不同。商標為單純之文字商標，或為文字結合圖形、數字等其他要素之聯合式商標，在近似程度之判斷上，可能因比對的主要部分或予人整體印象有所不同，而影響考量因素之比重，必須依個案具體情況而定。
※ A、B 常在同一句出現，兩者都講就各抽一點。例：「『功夫』二字外觀近似」→ 主要識別部分「共同含『功夫』」＋ 外觀「外觀近似」。
【粒度規則】
1. claim 要含案件具體內容（指名共同字、具體差異）。原文無細節就照原文，不要自行編造。
2. 綁在一起的多面向各抽一點。
3. claim 精簡成一句理由：不含註冊號（如「註冊第00799710號」）、不複製「本件商標與據以核駁…相較」之類框架，只留理由本身（如「兩商標共同含『天春』」）。
【範例1（真實案 T0398530；文字／差異／外觀／結論）】
輸入：商標是否近似暨其近似之程度商標近似係指二商標予人之整體印象有其相近之處，其判斷得就二商標外觀、觀念及讀音為觀察，查本件商標圖樣與據以核駁註冊第00799710號「鍋寶」商標、第00998744號「鍋寶」商標相較，皆有文字「鍋寶」，僅字體之些微差異，且皆予人單純「文字」商標之整體印象，在外觀、觀念、讀音上相雷同，以具有普通知識經驗之消費者，於購買時施以普通之注意，可能會誤認二商品來自同一來源或雖不相同但有關聯之來源，應屬構成近似之商標，且其近似程度高。
輸出：{"reason_points":[{"aspect":"主要識別部分","claim":"兩商標共同含文字『鍋寶』"},{"aspect":"差異","claim":"僅字體之些微差異"},{"aspect":"外觀","claim":"外觀相雷同"} ,{"aspect":"近似程度結論","claim":"構成近似之商標、近似程度高"}]}

【範例2（真實案 T0357231；主要識別部分/近似程度結論）】
輸入：商標是否近似暨其近似之程度商標近似係指二商標予人之整體印象有其相近之處，其判斷得就二商標外觀、觀念及讀音為觀察。經查，本件「馥蘭姬雪」商標圖樣，與據以核駁註冊第593422號「馥蘭皙兒」商標圖樣、註冊第1379455號「馥蘭皙兒深層極致」商標圖樣相較，二者分別以「馥蘭姬雪」、「馥蘭皙兒」為其商標主要識別部分，予人寓目印象均有醒目且深刻之「馥蘭」為字首；另與據以核駁註冊第1582957號「姬雪」商標圖樣、註冊第1501274號「姬雪GISELLE」商標圖樣相較，二者均有相同之「姬雪」二中文字，以具有普通知識經驗之消費者，於購買時施以普通之注意，仍有可能會誤認二商品來自同一來源或雖不相同但有關聯之來源，或誤認二商標之使用人間存在關係企業、授權關係、加盟關係或其他類似關係，應屬構成近似之商標，且其近似程度高。
輸出：{"reason_points":[{"aspect":"主要識別部分","claim":"與『馥蘭皙兒』商標均以醒目字首『馥蘭』為主要識別部分"},{"aspect":"主要識別部分","claim":"與『姬雪』商標均有相同之『姬雪』二字"} ,{"aspect":"近似程度結論","claim":"應屬構成近似之商標、近似程度高"}]}

【待拆解輸入】
```

The passage to decompose is appended after the final line.

## The four aspects

Both prompts use the same four values, taken from §5.2 of TIPO's Examination Guidelines on
Likelihood of Confusion and restricted to the factors decidable from the marks themselves. An
aspect outside this set is an error, not an unknown label.

| Value | Covers |
|---|---|
| `主要識別部分` | the salient element the two marks share |
| `差異` | the concrete distinguishing element |
| `外觀` | whether the marks look, sound or read alike |
| `近似程度結論` | whether similarity is found, and to what degree |
