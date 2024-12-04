from llama_api import LlamaApi

llama = LlamaApi(temperature=0.7, n_predict=1000)

text = """1. Mariage ou Pacs accompagné d'un changement de lieu de résidence
Bon à savoir : la démission doit se situer dans les 2 mois du mariage ou du Pacs (avant ou après). 	- Livret de famille, extrait ou copie de l’acte de mariage ou attestation d’inscription de la déclaration au greffe du tribunal judiciaire (avec noms, prénoms, date et lieu de naissance et date de l’enregistrement du Pacs).
- Un justificatif de domicile de l’ancien et du nouveau lieu de résidence (facture, bail…).
2. Démission pour suivre son conjoint qui change de lieu de résidence pour exercer un nouvel emploi salarié (ou non)
Bon à savoir : à la fin du contrat de travail suite à votre démission, inscrivez vous à l'agence France Travail de votre nouveau lieu de résidence, et non de celle de votre précédente résidence. Faute de quoi, vous ne pourrez pas faire valoir ce motif de démission légitime et être indemnisé.	- Qualité d’époux : copie du livret de famille, copie ou extrait de l’acte de mariage (moins de 12 mois), acte notarié récent ou le dernier avis d’imposition.
- Qualité de partenaire : attestation d’inscription récente (moins de 12 mois) de la déclaration au greffe du tribunal judiciaire (mention importante : noms, prénoms, date et lieu de naissance, date d’enregistrement du PACS) ou dernier avis d’imposition.
- Qualité de concubin : certificat de concubinage ou quittance de loyer ou tout autre justificatif de résidence de vie commune (les justificatifs doivent être antérieurs à la démission).
- Ordre de mutation ou contrat de travail ou bulletin de salaire ou attestation employeur ou extrait kbis ou inscription au CFE (pour les non salariés).
- Un justificatif de domicile de l’ancien et du nouveau lieu de résidence (facture, bail…).
3. Clause « de couple ou indivisible »	- Contrat de travail qui doit comporter une clause de résiliation automatique.
- Attestation employeur de l’autre titulaire du contrat afin de vérifier que le départ volontaire résulte du licenciement, d’une rupture conventionnelle ou de la mise à la retraite de ce dernier.
4. Mineur qui quitte son emploi pour suivre ses parents	- Un justificatif de domicile de l’ancien et du nouveau lieu de résidence des ascendants.
- Qualité de parent : s’il s’agit d’un tiers, document justifiant de l’autorité parentale.
5. Majeur "protégé" (sous tutelle, curatelle ou sauvegarde de justice) qui démissionne pour suivre son tuteur, curateur ou mandataire.	- La décision judiciaire désignant un « parent » comme mandataire spécial, tuteur ou curateur.
- Un justificatif de domicile de l’ancien et du nouveau lieu de résidence du majeur protégé.
- Un justificatif de domicile du « parent » mandataire spécial, tuteur ou curateur.
6. Enfant handicapé admis dans une structure d’accueil hors du lieu de résidence	- Livret de famille.
- Attestation de la structure d’accueil de la prise en charge de l’enfant handicapé.
- Un justificatif de domicile de l’ancien et du nouveau lieu de résidence (facture, bail…).
7. Victime de violences conjugales, imposant un changement de résidence	- Plainte déposée auprès du Procureur de la République.
- Citations directes devant le tribunal.
- Plainte avec constitution de partie civile devant le juge d’instruction.
- Plainte déposée auprès d’un commissariat ou d’une gendarmerie.
- Un justificatif de domicile de l’ancien et du nouveau lieu de résidence (facture, bail…).
8. Démission d'un nouveau contrat avant que ne se soient écoulés 65 jours travaillés, suite à un licenciement, une rupture conventionnelle ou une fin de CDD	- Attestation employeur.
- Attention : vous ne devez pas avoir été inscrit comme demandeur d’emploi entre cet avant-dernier emploi et la nouvelle période d’activité salariée rompue à votre initiative.
9. Démission après 3 années d’affiliation sans interruption, suivie d’un CDI auquel l'employeur met fin dans les 65 premiers jours travaillés	- Attestation employeur.
10. Echec dans la création ou la reprise d’une entreprise	- Immatriculation au répertoire des métiers.
- Déclaration au Centre de formalités des entreprises.
- Extrait K (personne physique) ou Kbis (personne morale).
- Preuve des difficultés de l'entreprise (difficultés financières attestées par un cabinet comptable, etc.).
11. L'employeur ne verse pas de salaire malgré une décision de justice	- Ordonnance de référé allouant une provision de sommes correspondant à des arriérés de salaires.
- Ordonnance condamnant l’employeur au versement d’une provision sur les salaires suite à l’introduction de sa demande devant le bureau de conciliation des prud’hommes.
- Jugement d’une juridiction prud'homale allouant une provision correspondant à des arriérés de salaire ou condamnant l’employeur au versement de créances salariales.
-Toute décision de justice condamnant l’employeur à verser les salaires à son salarié.
12. Victime d'un acte délictueux dans le cadre du contrat de travail	- Copie de la plainte ou récépissé de dépôt de celle-ci auprès du Procureur de la République.
- Citation directe (saisie directe du Tribunal de Police ou Correctionnel si contravention ou délit).
- Plainte déposée auprès d’un commissariat de police ou d’une gendarmerie.
13. Fin de contrat d’insertion par l’activité pour occuper un emploi ou une action de formation	- Contrat de travail de l’emploi repris.
- Attestation employeur de l’emploi repris.
- Attestation relative à la formation (entrée, présence).
14. Fin de contrat unique d’insertion (contrat d’accompagnement dans l’emploi pour le secteur non marchand, ou un contrat initiative emploi concernant le secteur marchand) pour un emploi en CDI ou CDD d’au moins 6 mois, ou pour suivre une action de formation qualifiante	- Contrat de travail de l’emploi repris.
- Attestation employeur de l’emploi repris.
 - Attestation relative à la formation (entrée, présence).
15. Suite à un contrat de service civique, ou de volontariat de solidarité internationale, ou de volontariat associatif (pour au moins un an)	- Attestation par l’association qui a engagé l’intéressé précisant la qualité de volontariat de solidarité internationale ou volontariat associatif d’une durée continue minimale d’un an.
- Attestation de service civique.
16. En tant que journaliste : suite à des problèmes de conscience professionnelle ou d’orientation politique	Attestation employeur comportant les précisions nécessaires.
17. En tant qu’assistant(e) maternel(e) : suite au refus de l'employeur de procéder aux vaccinations légales de son enfant	Lettre de démission ou attestation sur l’honneur mentionnant ce motif.
"""

translated = ""

for item in text.split(". "):
    
    to_translate = item + ". "
    prompt = """<|endoftext|><|im_start|>user
    Translate the following French text to Portuguese:
    $$TEXT$$
    <|im_start|>assistant
    """.replace("$$TEXT$$", to_translate)
    completetion_raw = llama.continue_text(prompt)
    completetion = completetion_raw['content'].strip()
    print(completetion)
    translated += completetion + "\n"


with open("translated.txt", mode="+w", encoding="utf-8") as f:
    f.write(translated)
