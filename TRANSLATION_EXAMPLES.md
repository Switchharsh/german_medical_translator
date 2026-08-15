# Translation examples — PARROT German radiology reports (DE→EN)

Verbatim output from benchmark jobs 3944003–3944005. `BLEU` is per-segment
sentence BLEU; `flags` lists which clinical detectors fired (neg = negation,
num = number/measurement, lat = laterality, term = terminology).

Segments are ordered by mean BLEU across the seven models, highest first.
Text is whitespace-normalised and truncated at 260 characters (`…`); nothing
else is altered.

| # | segment | mean BLEU | band |
|---|---|---|---|
| 1 | [parrot-1147](#parrot-1147) | 67 | good |
| 2 | [parrot-1154](#parrot-1154) | 66 | good |
| 3 | [parrot-1125](#parrot-1125) | 65 | good |
| 4 | [parrot-663](#parrot-663) | 64 | good |
| 5 | [parrot-597](#parrot-597) | 63 | good |
| 6 | [parrot-621](#parrot-621) | 62 | good |
| 7 | [parrot-1153](#parrot-1153) | 41 | okay |
| 8 | [parrot-650](#parrot-650) | 40 | okay |
| 9 | [parrot-626](#parrot-626) | 40 | okay |
| 10 | [parrot-651](#parrot-651) | 39 | okay |
| 11 | [parrot-648](#parrot-648) | 39 | okay |
| 12 | [parrot-619](#parrot-619) | 38 | okay |
| 13 | [parrot-604](#parrot-604) | 26 | poor |
| 14 | [parrot-606](#parrot-606) | 26 | poor |
| 15 | [parrot-1165](#parrot-1165) | 23 | poor |
| 16 | [parrot-581](#parrot-581) | 18 | poor |
| 17 | [parrot-609](#parrot-609) | 17 | poor |
| 18 | [parrot-580](#parrot-580) | 13 | poor |

---

<a id="parrot-1147"></a>
### 1. `parrot-1147` — mean BLEU 67 (good)

| | text |
|---|---|
| **German source** | Fragestellung: Verdacht auf Achillessehnenruptur Technik: Sonographie der Achillessehne rechts. Befund: Die Achillessehne ist im mittleren Drittel verdickt und zeigt eine Unterbrechung der Faserstruktur. Im Bereich der Ruptur findet sich ein Hämatom. Beurteilu… |
| **Reference (human)** | Clinical Question: Suspected Achilles tendon rupture Technique: Ultrasound of the right Achilles tendon. Findings: The Achilles tendon is thickened in the middle third and shows a disruption of the fiber structure. A hematoma is present at the site of the rupt… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 83 | — | Question: Suspected Achilles tendon rupture. Technique: Ultrasound of the right Achilles tendon. Findings: The Achilles tendon is thickened in the middle third and shows a disruption of the fiber structure. A hematoma is present in the area of the rupture. Ass… |
| translategemma-4b | 81 | — | Question: Suspected Achilles tendon rupture Technique: Ultrasound of the right Achilles tendon. Findings: The Achilles tendon appears thickened in the middle third and shows a disruption of the fiber structure. A hematoma is present in the area of the rupture.… |
| qwen35-4b | 68 | — | Question: Suspected rupture of the Achilles tendon. Technique: Ultrasound of the right Achilles tendon. Findings: The Achilles tendon is thickened in the middle third and shows a disruption of the fiber structure. In the area of the rupture, a hematoma is pres… |
| hymt2-30b-a3b | 76 | — | Question: Suspected rupture of the Achilles tendon Technique: Ultrasound examination of the right Achilles tendon. Findings: The Achilles tendon is thickened in its middle third and shows a disruption of the fiber structure. A hematoma is present at the site o… |
| hymt2-1.8b | 71 | — | Questionnaire: Suspected Achilles tendon rupture Technique: Ultrasound of the right Achilles tendon Findings: The Achilles tendon is thickened in the middle third and shows a disruption in its fiber structure. A hematoma is present at the site of the rupture. … |
| opus | 44 | — | Question wording: Suspect of Achilles tendon rupture Technique: Sonography of the Achilles tendon right. Findings: The Achilles tendon thickened in the middle third and shows a disruption of the fiber structure. In the area of the rupture there is a hematoma. |
| nllb | 48 | — | Technique: sonography of the Achilles tendon on the right. Findings: The Achilles tendon is thickened in the middle third and shows a disruption of the fiber structure. In the area of the rupture there is a hematoma. Assessment: Achilles tendon rupture on the … |

---

<a id="parrot-1154"></a>
### 2. `parrot-1154` — mean BLEU 66 (good)

| | text |
|---|---|
| **German source** | Technik: Röntgen Handgelenk links in zwei Ebenen (dorsopalmar und seitlich). Befund: Es zeigt sich eine Fraktur des distalen Radius mit Dislokation nach dorsal und radial. Die Fraktur reicht bis in die Gelenkfläche hinein. Das Ulna ist intakt. Beurteilung: Dis… |
| **Reference (human)** | Technique: X-ray of the left wrist in two views (posteroanterior and lateral). Findings: There is a fracture of the distal radius with dorsal and radial displacement. The fracture extends into the joint surface. The ulna is intact. Impression: Left distal radi… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 78 | — | Technique: X-ray of the left wrist in two planes (dorsopalmar and lateral). Findings: There is a fracture of the distal radius with dorsal and radial displacement. The fracture extends into the joint surface. The ulna is intact. Assessment: Distal radius fract… |
| translategemma-4b | 62 | — | Technique: X-ray of the left wrist, in two planes (dorsopalmar and lateral). Findings: A fracture of the distal radius is visible, with displacement towards the dorsal and radial sides. The fracture extends into the joint surface. The ulna is intact. Assessmen… |
| qwen35-4b | 80 | — | Technique: X-ray of the left wrist in two planes (dorsopalmar and lateral). Findings: There is a fracture of the distal radius with dorsal and radial displacement. The fracture extends into the articular surface. The ulna is intact. Assessment: Left distal rad… |
| hymt2-30b-a3b | 89 | — | Technique: X-ray of the left wrist in two planes (dorsopalmar and lateral). Findings: There is a fracture of the distal radius with dorsal and radial displacement. The fracture extends into the joint surface. The ulna is intact. Assessment: Left distal radius … |
| hymt2-1.8b | 56 | — | Technician: X-ray of the left wrist in two planes (dorsopalmar and lateral). Findings: A fracture of the distal radius is present, with displacement posteriorly and radially. The fracture extends into the joint surface. The ulna is intact. Assessment: Fracture… |
| opus | 45 | — | Technique: X-ray wrist left in two planes (dorsopalmar and side). Findings: A fracture of the distal radius with dislocation after dorsal and radial. The fracture extends into the joint surface. The ulna is intact. Assessment: Distal radius fracture left with … |
| nllb | 52 | — | Technique: X-ray left wrist in two planes (dorsopalmar and lateral). Findings: A fracture of the distal radius with dislocation to the dorsal and radial. The fracture extends into the joint surface. The ulna is intact. Assessment: Distal radius fracture left w… |

---

<a id="parrot-1125"></a>
### 3. `parrot-1125` — mean BLEU 65 (good)

| | text |
|---|---|
| **German source** | Fragestellung: V.a. akute Appendizitis Technik: CT-Abdomen mit Kontrastmittel in portalvenöser Phase. Befund: Die Appendix zeigt eine regelrechte Wanddicke und Kontrastmittelanreicherung. Kein periläsionales Ödem oder freie Flüssigkeit. Leber, Gallenblase, Pan… |
| **Reference (human)** | Clinical Question: Suspected acute appendicitis Technique: CT abdomen with contrast in the portal venous phase. Findings: The appendix exhibits normal wall thickness and contrast enhancement. No perilesional edema or free fluid. The liver, gallbladder, pancrea… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 72 | — | Question: Primarily, suspected acute appendicitis. Technique: CT scan of the abdomen with contrast agent in the portal venous phase. Findings: The appendix shows a normal wall thickness and contrast enhancement. No perilesional edema or free fluid. The liver, … |
| translategemma-4b | 73 | — | Question: Primarily suspected acute appendicitis Technique: CT scan of the abdomen with contrast agent in the portal venous phase. Findings: The appendix shows a normal wall thickness and contrast agent enhancement. No perilesional edema or free fluid. The liv… |
| qwen35-4b | 80 | — | Question: Suspected acute appendicitis. Technique: CT abdomen with contrast medium in the portal venous phase. Findings: The appendix shows normal wall thickness and contrast enhancement. No perilesional edema or free fluid. Liver, gallbladder, pancreas, splee… |
| hymt2-30b-a3b | 82 | — | Question: Especially acute appendicitis Technique: Abdominal CT with contrast during the portal venous phase. Findings: The appendix shows normal wall thickness and contrast enhancement. No perilesional edema or free fluid. The liver, gallbladder, pancreas, sp… |
| hymt2-1.8b | 62 | — | Questionnaire findings: Primarily acute appendicitis Technique: CT abdomen with contrast agent during portal venous phase. Findings: The appendix shows a normal wall thickness and increased contrast agent accumulation. No perilesional edema or free fluid. The … |
| opus | 38 | — | Question wording: Acute appendicitis Technique: CT-abdomen with contrast media in portalvenous phase. Findings: The appendix shows a real wall thickness and contrast media enrichment. No periletial oedema or free fluid. Liver, gallbladder, pancreas, spleen and… |
| nllb | 49 | — | Question: Acute appendicitis in particular Technique: CT abdomen with contrast agent in portal venous phase. Findings: The appendix shows a true wall thickness and contrast agent enrichment. No perilation edema or free fluid. Liver, gallbladder, pancreas, sple… |

---

<a id="parrot-663"></a>
### 4. `parrot-663` — mean BLEU 64 (good)

| | text |
|---|---|
| **German source** | Kein Nachweis konfluierender Infiltrate. Kein größerer Pleuraerguss oder Pneumothorax im Liegen. Herzsilhouette nicht höhergradig verbreitert. Keine akuten Zeichen der zentralvenösen Stauung. Fremdmaterial: Intakte Sternalcerclagen, Clips und Hautklammern. |
| **Reference (human)** | No evidence of confluent infiltrates. No significant pleural effusion or pneumothorax in the supine position. Cardiac silhouette is not significantly enlarged. No acute signs of central venous congestion. Foreign material: Intact sternal wires, clips, and skin… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 88 | — | No evidence of confluent infiltrates. No significant pleural effusion or pneumothorax in the supine position. Cardiac silhouette not significantly enlarged. No acute signs of central venous congestion. Foreign material: Intact sternal cerclages, clips, and ski… |
| translategemma-4b | 68 | — | No evidence of converging infiltrates. No significant pleural effusion or pneumothorax in the supine position. Heart silhouette not significantly widened. No acute signs of central venous congestion. Foreign material: Intact sternal clamps, clips, and skin sut… |
| qwen35-4b | 77 | — | No evidence of confluent infiltrates. No larger pleural effusion or pneumothorax in the supine position. Cardiac silhouette not significantly widened. No acute signs of central venous congestion. Foreign material: Intact sternal cerclages, clips, and skin stap… |
| hymt2-30b-a3b | 92 | — | No evidence of confluent infiltrates. No significant pleural effusion or pneumothorax in the supine position. Cardiac silhouette is not markedly widened. No acute signs of central venous congestion. Foreign material: Intact sternal wires, clips, and skin stapl… |
| hymt2-1.8b | 51 | — | No evidence of confluent infiltrates. No larger pleural effusion or pneumothorax during lying position. The heart silhouette is not significantly widened. No acute signs of central venous stasis. Foreign material: Intact sternal clips, clamps, and skin attachm… |
| opus | 49 | — | No evidence of confluent infiltrates. No major pleural effusion or pneumothorax in lying down. Heart silhouette not widened higher. No acute signs of central venous congestion. Foreign material: intact star alcerclagen, clips and skin clamps. |
| nllb | 25 | term | No evidence of confluent infiltrates, no major pleural effusion or lung in the lying position, no increased heart dilation, no acute signs of central venous congestion, foreign material, intact sternal cervical collagen, clips and skin clamps. |

---

<a id="parrot-597"></a>
### 5. `parrot-597` — mean BLEU 63 (good)

| | text |
|---|---|
| **German source** | Kein Pneumothorax. Kein Pleuraerguss. Kein Infiltrat. Unauffälliges Mediastinum. Normbreite Herzsilhouette. Kein pulmonalvenöser Stau. |
| **Reference (human)** | No pneumothorax. No pleural effusion. No infiltrate. Unremarkable mediastinum. Normal cardiac silhouette. No pulmonary venous congestion. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 100 | — | No pneumothorax. No pleural effusion. No infiltrate. Unremarkable mediastinum. Normal cardiac silhouette. No pulmonary venous congestion. |
| translategemma-4b | 63 | — | No pneumothorax. No pleural effusion. No infiltrates. Normal mediastinum. Normal-sized heart silhouette. No pulmonary venous congestion. |
| qwen35-4b | 79 | — | No pneumothorax. No pleural effusion. No infiltrate. Unremarkable mediastinum. Normal heart silhouette width. No pulmonary venous congestion. |
| hymt2-30b-a3b | 83 | — | No pneumothorax. No pleural effusion. No infiltrate. Unremarkable mediastinum. Normally sized cardiac silhouette. No pulmonary venous congestion. |
| hymt2-1.8b | 46 | — | No pneumothorax. No pleural effusion. No infiltration. Normal mediastinum appearance. Normal width of heart silhouette. No pulmonary venous stasis. |
| opus | 62 | — | No pneumothorax. No pleural effusion. No infiltrate. Inconspicuous mediastinal. Standard width of heart silhouette. No pulmonary venous congestion. |
| nllb | 7 | — | No pneumothorax, no pleural effusion, no infiltration, inconspicuous mediastinum, heart silhouette of normal width, no pulmonary venous congestion, and no signs of any other signs of trauma. |

---

<a id="parrot-621"></a>
### 6. `parrot-621` — mean BLEU 62 (good)

| | text |
|---|---|
| **German source** | Projektion der Spitze der Jejunalsonde paravertebral im linken Oberbauch. Keine freie Luft. Unspezifische Darmgasverteilung. Kein Ileus. |
| **Reference (human)** | Projection of the tip of the jejunal tube is paravertebral in the left upper abdomen. No free air. Nonspecific bowel gas distribution. No ileus. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 70 | — | Projection of the tip of the jejunal tube paravertebrally in the left upper abdomen. No free air. Nonspecific distribution of intestinal gas. No ileus. |
| translategemma-4b | 62 | — | Projection of the tip of the jejunal probe paravertebrally in the left upper abdomen. No free air. Non-specific distribution of intestinal gas. No ileus. |
| qwen35-4b | 76 | — | Projection of the tip of the jejunal tube paravertebrally in the left upper abdomen. No free air. Non-specific bowel gas distribution. No ileus. |
| hymt2-30b-a3b | 64 | — | Projection of the tip of the paravertebral jejunostomy tube in the left upper abdomen. No free air. Nonspecific distribution of intestinal gas. No ileus. |
| hymt2-1.8b | 59 | — | Projection of the tip of the jejunum probe in the paravertebral region of the left upper abdomen. No free air. Unspecific intestinal gas distribution. No ileus. |
| opus | 69 | — | Projection of the tip of the Jejun solstice paravertebral in the left upper abdomen. No free air. Non-specific intestinal gas distribution. No ileus. |
| nllb | 34 | neg-dropped | Projection of the tip of the jejunal probe paravertebral in the left upper abdomen. |

---

<a id="parrot-1153"></a>
### 7. `parrot-1153` — mean BLEU 41 (okay)

| | text |
|---|---|
| **German source** | Fragestellung: Schlaganfall? Technik: CT des Schädels nativ und mit Kontrastmittel. Befund: Im linken Hirnhemisphäre zeigt sich eine hypodens Läsion im Versorgungsgebiet der Arteria cerebri media, vereinbar mit einem ischämischen Hirninfarkt. Kein Hinweis auf … |
| **Reference (human)** | Clinical Question: Stroke? Technique: Non-contrast and contrast-enhanced CT of the head. Findings: There is a hypodense lesion in the left cerebral hemisphere within the territory of the middle cerebral artery, consistent with an ischemic cerebral infarction. … |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 57 | — | Question: Stroke? Technique: Native and contrast-enhanced CT scan of the skull. Findings: In the left cerebral hemisphere, a hypodense lesion is visible in the territory supplied by the middle cerebral artery, consistent with an ischemic cerebral infarction. N… |
| translategemma-4b | 24 | — | Question: Stroke? Technique: CT scan of the skull, both without and with contrast agent. Findings: In the left hemisphere of the brain, a hypodense lesion is visible in the area supplied by the arteria cerebri media, which is consistent with an ischemic brain … |
| qwen35-4b | 76 | — | Question: Stroke? Technique: Non-contrast and contrast-enhanced CT of the head. Findings: In the left cerebral hemisphere, a hypodense lesion is visible in the territory of the middle cerebral artery, consistent with an ischemic cerebral infarction. No evidenc… |
| hymt2-30b-a3b | 51 | — | Question: Stroke? Technique: Non-contrast and contrast-enhanced CT of the skull. Findings: In the left cerebral hemisphere, there is a hypodense lesion in the territory supplied by the middle cerebral artery, consistent with an ischemic infarct. No signs of he… |
| hymt2-1.8b | 40 | — | Question: Stroke? Technique: CT scan of the skull, with contrast agent. Findings: In the left cerebral hemisphere, there is a hypodense lesion in the territory of the arteria cerebri media, consistent with an ischemic stroke. No signs of bleeding. Assessment: … |
| opus | 15 | — | Question wording: Stroke? Technique: CT of the skull natively and with contrast agent. Findings: In the left hemisphere of the brain a hypodens lesions in the supply area of the arteria cerebri media, compatible with an ischaemic brain infarction. No indicatio… |
| nllb | 21 | — | Question: Stroke? CT of the skull native and contrast material. Findings: Hypodic lesion in the left hemisphere of the brain, in the supply area of the arteries cerebri media, consistent with ischemic stroke. No evidence of bleeding. Assessment: Ischemic strok… |

---

<a id="parrot-650"></a>
### 8. `parrot-650` — mean BLEU 40 (okay)

| | text |
|---|---|
| **German source** | Z.n. Voralarm. Z.n. Verkehrsunfall. Re-FAST erbeten. Befund / Beurteilung: Polytrauma-CT zum Vergleich vorliegend. Kein Nachweis freier Flüssigkeit im Morison- und Koller-Pouch, sowie im Douglas-Raum. Kein Perikarderguss |
| **Reference (human)** | Status post pre-alert. Status post traffic accident. Re-FAST requested. Findings / Assessment: Polytrauma CT available for comparison. No evidence of free fluid in the Morison's pouch, Koller pouch, or Douglas pouch. No pericardial effusion. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 41 | — | Regarding the preliminary alarm. Regarding the traffic accident. Requesting a repeat FAST scan. Findings/Assessment: A previous polytrauma CT scan is available for comparison. No evidence of free fluid in the Morrison pouch, the paracolic gutter, or the Dougla… |
| translategemma-4b | 37 | — | Following a prior alarm. Following a traffic accident. Request for re-FAST. Findings/Assessment: CT scan of the whole body available for comparison. No evidence of free fluid in the Morrison pouch and the Koller pouch, as well as in the Douglas space. No peric… |
| qwen35-4b | 59 | — | History: Pre-alert. Traffic accident. Re-FAST requested. Findings/Assessment: Polytrauma CT available for comparison. No free fluid detected in the Morison's pouch, Koller's pouch, or Douglas space. No pericardial effusion. |
| hymt2-30b-a3b | 32 | — | Note: Early warning. Note: Traffic accident. Re-FAST requested. Findings/Assessment: Comparative polytrauma CT is available. No free fluid detected in Morison’s pouch, Coller’s pouch, or Douglas’ space. No pericardial effusion |
| hymt2-1.8b | 53 | — | Z.n. Voralarm. Z.n. Traffic accident. Re-FAST requested. Findings/Assessment: Polytrauma CT available for comparison. No evidence of free fluid in the Morison and Koller pouch, as well as in the Douglas space. No pericardial effusion. |
| opus | 40 | term | For example, pre-alarm. For example, traffic accident. Re-FAST requested. Findings / Assessment: Polytrauma-CT for comparison available. No evidence of free fluid in the morison and collery, as well as in the Douglas room. No pericardial casting |
| nllb | 17 | term | Preliminary alert, accident, Re-FAST requested, finding/assessment: polytrauma CT for comparison, no evidence of free fluid in the Morrison and Koller pouch, and in the Douglas chamber, no pericardial casting |

---

<a id="parrot-626"></a>
### 9. `parrot-626` — mean BLEU 40 (okay)

| | text |
|---|---|
| **German source** | Rippenserienfraktur der 8.-11. Rippe rechts. Kein Pneumothorax. Kein Pleuraerguss. Geringe Belüftungsstörungen im rechten Unterfeld. Mittelständiges Mediastium. Normbreite Herzsilhouette. Kein PV-Stau. |
| **Reference (human)** | Fracture of the 8th to 11th ribs on the right side. No pneumothorax. No pleural effusion. Mild ventilation disturbances in the right lower lung field. Mediastinum is centrally positioned. Heart silhouette is of normal wiRXh. No pulmonary venous congestion. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 69 | — | Serial rib fractures of the 8th to 11th ribs on the right side. No pneumothorax. No pleural effusion. Mild ventilation disturbances in the right lower lung field. Mediastinum of normal size. Heart silhouette of normal width. No signs of pulmonary venous conges… |
| translategemma-4b | 50 | — | Fracture of the 8th to 11th ribs on the right side. No pneumothorax. No pleural effusion. Mild respiratory impairment in the lower right field. Moderately sized mediastinum. Normal-sized heart silhouette. No signs of pulmonary congestion. |
| qwen35-4b | 53 | — | Fractures of the 8th to 11th ribs on the right. No pneumothorax. No pleural effusion. Mild ventilation impairment in the right lower field. Intermediate-sized mediastinum. Normal heart silhouette width. No pulmonary venous congestion. |
| hymt2-30b-a3b | 54 | — | Fractures of the 8th–11th ribs on the right side. No pneumothorax. No pleural effusion. Mild ventilation abnormalities in the right lower lung field. Midline mediastinum. Normally sized cardiac silhouette. No pulmonary venous congestion. |
| hymt2-1.8b | 22 | — | Right 8th–11th rib fracture. No pneumothorax. No pleural effusion. Minor ventilation disorders in the right lower field. Moderate mediastinum. Normal chest silhouette. No PV obstruction. |
| opus | 21 | — | Rib fracture of the 8th-11th rib right. No pneumothorax. No pleural effusion. Low ventilation problems in the right subfield. Medium-sized mediastium. Standard width heart silhouette. No PV-stau. |
| nllb | 10 | — | Ripper series fracture of the 8th-11th rib on the right, no pneumothorax, no pleural effusion, slight ventilation problems in the right lower quadrant, median mediastinum, normal heart shape, no PV congestion. |

---

<a id="parrot-651"></a>
### 10. `parrot-651` — mean BLEU 39 (okay)

| | text |
|---|---|
| **German source** | Z.n. Voralarm. Z.n. Verkehrsunfall. Re-FAST erbeten. Befund / Beurteilung: Polytrauma-CT zum Vergleich vorliegend. Ausgeprägte Flüssigkeitskollektion im Morsion- und Koller-Pouch, sowie im Douglas Raum. Kein Perikarderguss. Zügige Abklärung empfohlen, telefoni… |
| **Reference (human)** | Status post pre-alert. Status post traffic accident. Re-FAST requested. Findings / Assessment: Polytrauma CT available for comparison. Significant fluid collection in the Morison's pouch, Koller pouch, and Douglas space. No pericardial effusion. Prompt evaluat… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 36 | — | Regarding: Preliminary alert. Regarding: Traffic accident. Re-FAST examination requested. Findings/Assessment: CT scan for polytrauma available for comparison. Pronounced fluid collection in the Morrison and Cul-de-sac pouches, as well as in the Douglas space.… |
| translategemma-4b | 37 | — | Following a prior alarm. Following a traffic accident. Re-FAST requested. Findings/Assessment: A CT scan for comparison is available. Significant fluid collection in the Morrison and Coller pouches, as well as in the Douglas space. No pericardial effusion. A p… |
| qwen35-4b | 59 | — | History: Pre-alert. Traffic accident. Re-FAST requested. Findings/Assessment: Polytrauma CT available for comparison. Marked fluid collection in the Morison's pouch and the pouch of Douglas. No pericardial effusion. Prompt evaluation recommended; telephone con… |
| hymt2-30b-a3b | 48 | — | Preliminary alert. Traffic accident. Re-FAST requested. Findings/Assessment: A CT scan for polytrauma is available for comparison. Significant fluid accumulation in the Morison and Coller pouches, as well as in the Douglas space. No pericardial effusion. Promp… |
| hymt2-1.8b | 46 | — | Z.n. Voralarm. Z.n. Traffic accident. Re-FAST requested. Findings/Assessment: Polytrauma CT available for comparison. Marked fluid collection in the Morsion and Coller pouch, as well as in the Douglas space. No pericardial effusion. Rapid investigation recomme… |
| opus | 29 | term | For example, pre-alarm. For example, traffic accident. Re-FAST requested. Findings / Assessment: Polytrauma-CT for comparison available. Expressed liquid collection in the Morsion and Koller-Pouch, as well as in the Douglas room. No pericardial casting. Modera… |
| nllb | 18 | term | The following information is provided in the following sections: Preliminary alert, Traffic accident, Re-FAST requested, Conclusion/Assessment: Polytrauma CT for comparison, Prominent fluid collection in the Morsion and Koller pouch, and in the Douglas chamber… |

---

<a id="parrot-648"></a>
### 11. `parrot-648` — mean BLEU 39 (okay)

| | text |
|---|---|
| **German source** | Z.n. NTx. Perfusion? RIs? Verhalt? Nierentransplantat: Befund / Beurteilung: Transplantatniere morphologisch unauffällig, Polabstand 10.5 cm. Kein perirenaler Verhalt. Kleines Hämatom am apikalen Nierenpol (< 1cm). Regelrechte Organperfusion in der FKDS. Paren… |
| **Reference (human)** | Status post kidney transplant. Perfusion? RIs? Retention? Kidney Transplant: Findings / Assessment: The transplanted kidney appears morphologically normal, with a pole-to-pole length of 10.5 cm. No perirenal fluid collection. Small hematoma at the apical pole … |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 58 | num | Post-NTx. Perfusion? Resistive Index (RI)? Any abnormalities? Kidney transplant: Findings/Assessment: The transplanted kidney appears morphologically normal, with a polar distance of 10.5 cm. No perirenal abnormalities. Small hematoma at the apical pole of the… |
| translategemma-4b | 37 | num | Following NTx (Nephro-Tonsil exchange)? RIs (renal inflow)? Behavior? Kidney transplant: Findings/Assessment: The transplanted kidney appears morphologically normal, with a distance between the poles of 10.5 cm. There is no abnormality around the kidney. A sma… |
| qwen35-4b | 39 | num | Preoperatively: NTx. Perfusion? Resistances? Behavior? Kidney transplant: Findings / Assessment: Transplant kidney morphologically unremarkable, pole-to-pole distance 10.5 cm. No perirenal hematoma. Small hematoma at the apical renal pole (< 1 cm). Normal orga… |
| hymt2-30b-a3b | 45 | num | Z.n. NTx. Perfusion? RIs? Findings? Kidney transplant: Findings/Assessment: The transplanted kidney appears morphologically normal; pole-to-pole distance is 10.5 cm. No perirenal findings. A small hematoma at the renal apex (<1 cm). Normal organ perfusion on D… |
| hymt2-1.8b | 39 | num | Z.n. NTx. Perfusion? RIs? Renal behavior? Kidney transplant: Findings/Assessment: The transplanted kidney is morphologically normal; renal pelvis width is 10.5 cm. No perirenal behavior is observed. A small hematoma is present at the apex of the kidney pelvis … |
| opus | 29 | — | NTx. Perfusion? RIs? Relation? Renal transplant: Findings / Evaluation: Transplant kidneys morphologically inconspicuous, pole spacing 10.5 cm. No perirenal retention. Small hematoma at the apical kidney pole (< 1cm). Regulated organ perfusion in the FKDS. Par… |
| nllb | 24 | — | NTx. Perfusion? RI? Behavior? Kidney transplant: finding/ assessment: graft recipients morphologically unremarkable, pole distance 10.5 cm. No perirenal behavior. Small hematoma at apical renal pole (< 1 cm). Regular organ perfusion in the FKDS. Parenchyma RIs… |

---

<a id="parrot-619"></a>
### 12. `parrot-619` — mean BLEU 38 (okay)

| | text |
|---|---|
| **German source** | Ca. 5 x 2 cm messender, länglicher röntgendichter Fremdkörper in Projektion auf das Rektum. Keine freie Luft. Kein Ileus. Unspezifische Darmgasverteilung. |
| **Reference (human)** | Approximately 5 x 2 cm elongated, radiopaque foreign body projected over the rectum. No free air. No signs of ileus. Nonspecific bowel gas pattern. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 44 | — | An approximately 5 x 2 cm, elongated, radiopaque foreign body is visible in projection onto the rectum. No free air. No ileus. Nonspecific distribution of intestinal gas. |
| translategemma-4b | 30 | — | Approximately 5 x 2 cm in size, a long, radiopaque foreign object is visible in projection on the rectum. No free air is present. No ileus. Non-specific distribution of bowel gas. |
| qwen35-4b | 60 | — | Approximately 5 x 2 cm, elongated radiopaque foreign body projected over the rectum. No free air. No ileus. Non-specific bowel gas distribution. |
| hymt2-30b-a3b | 45 | — | An elongated, radiopaque foreign body measuring approx. 5 x 2 cm, projected onto the rectum. No free air. No ileus. Nonspecific distribution of intestinal gas. |
| hymt2-1.8b | 44 | — | Approximately 5 x 2 cm in size, elongated radiolucent foreign body projected onto the rectum. No free air. No ileus. Unspecific intestinal gas distribution. |
| opus | 36 | — | Approx. 5 x 2 cm measuring, elongated X-ray-proof foreign body in projection on the rectum. No free air. No ileus. Non-specific intestinal gas distribution. |
| nllb | 8 | num | About five by two centimeters, elongated X-ray-detected foreign body projected onto the rectum, no free air, no ileus, nonspecific intestinal gas distribution. |

---

<a id="parrot-604"></a>
### 13. `parrot-604` — mean BLEU 26 (poor)

| | text |
|---|---|
| **German source** | Keine freie Luft. Einzelne Dünndarmschlingen deutlich dilatiert, möglicher Hinweis auf ein mechanisches Passagehindernis, CT empfohlen. |
| **Reference (human)** | No free air. Several small bowel loops are significantly dilated, possibly indicating a mechanical obstruction. A CT scan is recommended. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 29 | — | No free air. Individual loops of the small intestine are clearly dilated, which may indicate a mechanical obstruction; a CT scan is recommended. |
| translategemma-4b | 30 | — | No free space. Several small loops in the small intestine are noticeably dilated, which may indicate a mechanical obstruction. CT scan is recommended. |
| qwen35-4b | 41 | — | No free air. Individual loops of the small intestine are clearly dilated, possibly indicating a mechanical obstruction; CT scan recommended. |
| hymt2-30b-a3b | 27 | — | No free air. Individual loops of small intestine are markedly dilated; this may indicate a mechanical obstruction. CT is recommended. |
| hymt2-1.8b | 30 | — | No free air. Individual small intestinal loops are clearly dilated; this may indicate a mechanical obstruction in passage. CT scan is recommended. |
| opus | 16 | — | No free air. Individual small intestine slings clearly dilated, possible indication of a mechanical passage obstruction, CT recommended. |
| nllb | 10 | — | No free air, individual small intestinal tracts markedly dilated, possible indication of mechanical obstruction of passage, CT recommended. |

---

<a id="parrot-606"></a>
### 14. `parrot-606` — mean BLEU 26 (poor)

| | text |
|---|---|
| **German source** | Keine freie Luft. Keine Hinweis auf einen Ileus. Unspezifische Darmgasverteilung. Koprolithen und stuhlgefühltes Rektum. |
| **Reference (human)** | No free air. No evidence of an ileus. Nonspecific bowel gas distribution. Fecaliths and stool-filled rectum. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 33 | — | No free air. No signs of an ileus. Nonspecific distribution of intestinal gas. Coproliths and a rectum filled with stool. |
| translategemma-4b | 21 | — | No free air. No indication of ileus. Non-specific distribution of intestinal gas. Presence of fecal stones and a stool-like rectum. |
| qwen35-4b | 42 | — | No free air. No indication of an ileus. Non-specific distribution of intestinal gas. Coproliths and stool-filled rectum. |
| hymt2-30b-a3b | 25 | — | No free air. No signs of ileus. Nonspecific distribution of intestinal gas. Coprolites and rectum with stool-like material. |
| hymt2-1.8b | 10 | — | No free air in the abdomen. No signs of ileus. Unspecific distribution of intestinal gas. Kopliites and rectal sensation during defecation. |
| opus | 39 | — | No free air. No indication of an ileus. Non-specific intestinal gas distribution. Coprolites and chair-feeling rectum. |
| nllb | 11 | — | No open air, no sign of an ileus, nonspecific intestinal gas distribution, coprolites and a stool-like rectum. |

---

<a id="parrot-1165"></a>
### 15. `parrot-1165` — mean BLEU 23 (poor)

| | text |
|---|---|
| **German source** | Verlaufskontrolle nach Osteosynthese zur Refixierung Capitulum humeri. Zur intraoperativen Dokumentation Vom 24.6. unverändert kanülierte Schraube mit Unterlegscheibe distaler Humerus zur Refixierung Capitulum humeri, soweit erkenntlich fortschreitende Konsoli… |
| **Reference (human)** | Follow-up after osteosynthesis for refixation of the capitulum humeri. For intraoperative documentation from June 24th, the cannulated screw with washer in the distal humerus for refixation of the capitulum humeri remains unchanged. As far as can be determined… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 38 | — | Follow-up check after osteosynthesis for refixation of the humeral head. For intraoperative documentation: As of June 24, the previously inserted screw with washer in the distal humerus remains unchanged, used for refixation of the humeral head. As far as can … |
| translategemma-4b | 22 | — | Post-operative check after osteosynthesis for the refixation of the humerus head. For intraoperative documentation. From June 24th, the cannula-inserted screw with a washer on the distal humerus, used for refixation of the humerus head, remains unchanged, with… |
| qwen35-4b | 31 | — | Follow-up after osteosynthesis for fixation of the humeral capitulum. Intraoperative documentation: As of June 24, the cannulated screw with washer in the distal humerus for capitulum humeri fixation remains unchanged where visible, with progressive consolidat… |
| hymt2-30b-a3b | 28 | — | Follow-up after osteosynthesis for reattachment of the humeral capitulum. For intraoperative documentation: As of 24.6, the cannulated screw with washer remains unchanged at the distal humerus for reattachment of the humeral capitulum; apparently, consolidatio… |
| hymt2-1.8b | 15 | — | Course control after osteosynthesis for the refixation of the humeral capitulum. For intraoperative documentation: On June 24th, a unchanged cannulated screw with a lower bushing was inserted distally in the humerus for the refixation of the humeral capitulum.… |
| opus | 14 | — | Process control after osteosynthesis for refixing Capitulum humeri. For intraoperative documentation From 24.6. unchanged cannulaled screw with washer distal humerus for refixing Capitulum humeri, if recognized progressive consolidation, appropriate position. |
| nllb | 11 | num | For intraoperative documentation From 24 June, unchanged cannulated screw with a discal support for the humerus, for the fixation of the capitulum humeri, in so far as progressive consolidation is detectable, in axial position. |

---

<a id="parrot-581"></a>
### 16. `parrot-581` — mean BLEU 18 (poor)

| | text |
|---|---|
| **German source** | Zwischenzeitlich neu einbrachte Osteosynthese von Lendenwirbelkörper 1 auf Lendenwirbelkörper 3 bei bekannter mehrfragmentärer Fraktur von Lendenwirbelkörer 2. Postoperativ geringes umgebendes Weichteilhämatom im Zugangsweg. Keine Materialkomplikationen. Regel… |
| **Reference (human)** | Interim newly inserted osteosynthesis from lumbar vertebra 1 to lumbar vertebra 3 for a known comminuted fracture of lumbar vertebra 2. Postoperative minor surrounding soft tissue hematoma along the access path. No material complications. Proper screw position… |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 21 | num | In the meantime, a new osteosynthesis was performed, connecting lumbar vertebra 1 to lumbar vertebra 3, in a patient with a known multi-fragmentary fracture of lumbar vertebra 2. Postoperatively, a small surrounding soft tissue hematoma was observed along the … |
| translategemma-4b | 8 | — | Interim, a new osteosynthesis of vertebral body 1 onto vertebral body 3 was performed, given a known multi-fragmentary fracture of vertebral body 2. A small, localized hematoma of surrounding soft tissue was present at the surgical site. No material complicati… |
| qwen35-4b | 22 | — | Interim new osteosynthesis of the lumbar vertebral body 1 to the lumbar vertebral body 3 in the setting of a known multifragmentary fracture of the lumbar vertebral body 2. Postoperatively, a small surrounding soft tissue hematoma at the access site. No materi… |
| hymt2-30b-a3b | 21 | — | Repeated osteosynthesis between lumbar vertebra 1 and lumbar vertebra 3 in a patient with a known multifragmentary fracture of lumbar vertebra 2. Postoperatively, there was a small perifocal soft tissue hematoma along the surgical approach. No material-related… |
| hymt2-1.8b | 10 | — | In the meantime, an osteosynthesis of the lumbar spine from segment 1 to segment 3 was performed for a known multi-fragmented fracture of the lumbar spine segment 2. Postoperatively, there was minimal soft tissue hematoma in the access site. No material compli… |
| opus | 23 | — | In the meantime, osteosynthesis of lumbar vertebrae 1 on lumbar vertebrae 3 was introduced in the case of known multi-fragmentary fracture of lumbar vertebrae 2. Postoperatively low surrounding soft tissue hematoma in the access path. No material complications… |
| nllb | 17 | — | In the meantime, newly introduced osteosynthesis from lumbar vertebrae 1 to lumbar vertebrae 3 with known multi-fragmentary fracture of lumbar vertebrae 2. postoperatively small surrounding soft hematochore in the access pathway. No material complications. Reg… |

---

<a id="parrot-609"></a>
### 17. `parrot-609` — mean BLEU 17 (poor)

| | text |
|---|---|
| **German source** | Nach Repositionsversuch persistierende ventrale Schulterluxation rechts, kein Hinweis auf eine Begleitfraktur in loco typico. |
| **Reference (human)** | After an attempted reduction, there is a persistent anterior dislocation of the right shoulder, with no indication of an associated fracture in the typical location. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 37 | — | Following an attempted repositioning, a persistent anterior shoulder dislocation on the right side was observed; there was no evidence of an associated fracture in the typical location. |
| translategemma-4b | 19 | — | Following a repositioning attempt, persistent right-sided shoulder dislocation; no evidence of a concurrent fracture in the typical location. |
| qwen35-4b | 25 | — | Persistent ventral shoulder dislocation on the right following reduction attempt, no indication of a concomitant fracture in the typical location. |
| hymt2-30b-a3b | 22 | — | Right ventral shoulder dislocation persisting after reduction attempt; no signs of associated fracture in the typical location. |
| hymt2-1.8b | 6 | — | After attempts at repositioning, a persistent ventral shoulder dislocation on the right side occurred; no evidence of associated fracture at the site. |
| opus | 7 | — | After repositioning attempt persistent ventral shoulder luxation right, no indication of a follow-up fracture in loco typico. |
| nllb | 4 | — | After repositioning, persistent right ventral shoulder dislocation, no evidence of a concomitant fracture in loco typico. |

---

<a id="parrot-580"></a>
### 18. `parrot-580` — mean BLEU 13 (poor)

| | text |
|---|---|
| **German source** | Neu abgrenzbare Deckplattenimpression LWK 3 mit dorsaler Stempelbildung bei bekannter Osteoporose. Darüberhinaus regelrechtes Hinterkantenalignement. Geringe degenerative Wirbelsäulenveränderungen. |
| **Reference (human)** | Newly demarcated endplate impression at L3 with posterior indentation formation in the context of known osteoporosis. Otherwise, normal posterior alignment. Mild degenerative changes in the spine. |

| model | BLEU | flags | translation |
|---|---|---|---|
| translategemma-27b | 14 | — | Newly definable endplate impression of the LWK 3 with dorsal stamp formation in a patient with known osteoporosis. Furthermore, proper posterior edge alignment. Minimal degenerative changes in the vertebral column. |
| translategemma-4b | 16 | — | New, clearly distinguishable impression of LWK 3 (lumbar vertebral body impression) with dorsal impression, suitable for patients with known osteoporosis. Also, precise alignment of the posterior edge. Minimal degenerative changes in the spine. |
| qwen35-4b | 19 | — | Newly distinguishable lumbar vertebral body impression LWK 3 with dorsal stamp formation in the setting of known osteoporosis. Additionally, there is a clear posterior edge alignment. Mild degenerative spinal changes. |
| hymt2-30b-a3b | 10 | — | Newly distinguishable LWK 3 impression of the cover plates with dorsal stamping in a patient with known osteoporosis. Additionally, normal alignment of the posterior edges. Minimal degenerative changes of the spine. |
| hymt2-1.8b | 22 | — | Neu-isolated deck plate impression LWK 3 with a dorsal stempel formation in known osteoporosis. In addition, there is a proper posterior edge alignment. Minimal degenerative changes in the spine. |
| opus | 7 | — | Newly delimitable top plate impression LWK 3 with dorsal stamp formation with known osteoporosis. Moreover, real rear edge alignment. Low degenerative spinal changes. |
| nllb | 4 | — | New delimitable LWK 3 cover plate print with dorsal stamping in known osteoporosis, plus straight rear edge alignment, minor degenerative spinal changes. |

