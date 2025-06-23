# Importer nødvendige Python- og machine learning-biblioteker
import os  # Til at arbejde med filsystemet
import sys  # Til at stoppe programmet med sys.exit()
import random  # Til tilfældige tal og seed
import numpy as np  # NumPy bruges til tal og arrays
import pandas as pd  # Pandas bruges til dataframes (rækker og kolonner)
import torch  # PyTorch bruges til at bygge og træne AI-modellen

# Henter hjælpemoduler fra PyTorch
from torch.utils.data import Dataset, DataLoader  # Dataset- og batch-håndtering
from torch.optim import AdamW  # Optimizer til at opdatere modellen

# Henter komponenter fra Transformers-biblioteket (HuggingFace BERT)
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup

# Scikit-learn bruges til dataopdeling og evaluering
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Slår TensorFlow-logning fra, så output bliver renere
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Funktion: Generér eksempel-data hvis der ikke findes en CSV-fil
def generate_sample_data(path):
    # Lav en liste af beskeder med tilhørende labels (0 = sikker, 1 = mistænkelig)
    sample = [
        {"text": "Hej, hvordan går det?", "label": 0},
        {"text": "Skal vi mødes til kaffe i morgen?", "label": 0},
        {"text": "Reminder: dit møde er kl. 14:00", "label": 0},
        {"text": "Kan du sende mig rapporten inden fredag?", "label": 0},
        {"text": "Tak for hjælpen!", "label": 0},
        {"text": "Hvad synes du om filmen?", "label": 0},
        {"text": "Vi ses snart", "label": 0},
        {"text": "Må jeg låne din bil til lørdag?", "label": 0},
        {"text": "Jeg har våben til salg, privat besked for pris", "label": 1},
        {"text": "Send koden til din adgangskort straks", "label": 1},
        {"text": "Mød mig bag biografen kl. 23 uden vidner", "label": 1},
        {"text": "Tror du på hurtig profit fra narkohandel?", "label": 1},
        {"text": "Jeg har stoffer klar, betal 1000 DKK før levering", "label": 1},
        {"text": "Du skal betale eller vi gør det afslut", "label": 1},
        {"text": "Giv mig adgang til din konto for betaling", "label": 1},
        {"text": "Vi får fuld kontrol over systemet i aften", "label": 1}
    ]    # sample: list of dict[str, int]
    # Gemmer listen som en CSV-fil med pandas
    pd.DataFrame(sample).to_csv(path, index=False)
    print(f"Generated sample data.csv med {len(sample)} rækker")

# Hvis filen ikke eksisterer eller er tom, lav en ny
csv_path = 'data.csv'
if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    generate_sample_data(csv_path)

# Funktion: Sæt seed for at få samme resultater hver gang           
def set_seed(seed: int = 42):
    random.seed(seed)   # sætter seed for Python random (ingen return)
    np.random.seed(seed)    # sætter seed for NumPy random
    torch.manual_seed(seed)   # sætter seed for PyTorch CPU random
    torch.cuda.manual_seed_all(seed)    # sætter seed for PyTorch GPU random (hvis GPU tilgængelig)

# Funktion: Læs CSV, tokenizér tekst og opret data loaders                      
# create_data_loaders returnerer: train_loader, val_loader (tuple of DataLoader)
def create_data_loaders(path, tokenizer, max_length=128, batch_size=16, test_size=0.1):
    # Læs CSV-fil, # Læs CSV til pandas.DataFrame
    df = pd.read_csv(path)  # df: pandas.DataFrame
    df = df.dropna().reset_index(drop=True)  # Fjern tomme rækker

    # Split data i træning og validering
    # stratify=df['label'] sikrer samme label-fordeling i begge sæt
    # Del data op i træning og validering                               
    train_df, val_df = train_test_split(    #splitter op, tuple bliver pakket ud
        df, test_size=test_size, stratify=df['label'], random_state=42)
        # train_df, val_df: pandas.DataFrame

    # Klasse: Definerer hvordan data læses et element ad gangen                 
    class CrimeDataset(Dataset):
        def __init__(self, texts, labels):
            # texts: pandas.Series med str; labels: pandas.Series med ints
            self.texts = texts.tolist() # konverter til list[str]
            self.labels = labels.tolist()   # konverter til list[int]

        def __len__(self):
            # Returnerer antal eksempler (int)
            return len(self.texts)  

        def __getitem__(self, idx):                                      
            # Konverter tekst til tokens + maskering
            encoding = tokenizer.encode_plus(
                self.texts[idx],             # Henter teksten på plads idx i listen
                add_special_tokens=True,    # Tilføjer specielle tokens ([CLS], [SEP]) som BERT kræver
                max_length=max_length,       # Sikrer at alle tekster har samme længde
                padding='max_length',        # Padder korte tekster op til max_length
                truncation=True,            # Afkorter lange tekster til max_length
                return_attention_mask=True,     # Laver en attention mask (1 for rigtige tokens, 0 for padding)
                return_tensors='pt'     # Returnerer resultatet som PyTorch tensors
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),       # Tensor med token-id'er
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)  
            }

    # Opret trænings- og valideringsdatasæt                     
    train_ds = CrimeDataset(train_df['text'], train_df['label'])    # train_ds: Dataset
    val_ds = CrimeDataset(val_df['text'], val_df['label'])          # val_ds: Dataset

    # Opret data loaders til brug i træningen                           
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)    #DataLoader er en klasse i PyTorch
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader     # return: tuple(DataLoader, DataLoader)

# Funktion: Træn modellen i én epoch                  
def train_epoch(model, loader, optimizer, scheduler, device):       
    model.train()  # Sæt model til træningstilstand
    losses = []     # Python list[float]

    for batch in loader:  # For hver mini-batch        
        # Flyt data til valgt device (GPU eller CPU)
        ids = batch['input_ids'].to(device)         #device hvor foregår processen gpu og cpu
        mask = batch['attention_mask'].to(device)   #nøgler 
        labels = batch['labels'].to(device)       # torch.Tensor, shape [batch_size]
 
        # Forudsigelse og beregning af loss                            
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())

        # Bagudpropagering og opdatering af vægte
        optimizer.zero_grad()   # Nulstil tidligere gradienter           
        loss.backward()         # Tilbagepropagation også # beregn gradienter
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # Undgå eksploderende gradienter
        optimizer.step()        # Opdater modelparametre
        scheduler.step()        # Opdater læringsrate

    return np.mean(losses)  # returnerer gennemsnitligt loss som float

# Funktion: Evaluer modellen på valideringsdata         
def eval_model(model, loader, device):
    model.eval()  # Sæt model i evaluerings-tilstand
    preds, true = [], []    # Python lists til at samle resultater

# Slå læring fra (hurtigere evaluering)
    with torch.no_grad():  # type context manager i pytorch. # Deaktiver gradientberegning      
        for batch in loader:
            ids = batch['input_ids'].to(device)          # Flytter input_ids (tokeniserede tekster)
            mask = batch['attention_mask'].to(device)   ##fortæller hvilke tokens der er padding
            labels = batch['labels'].to(device)

            outputs = model(input_ids=ids, attention_mask=mask) # Kører modellen fremad (uden labels, så kun logits returneres)
            probs = torch.softmax(outputs.logits, dim=1)     # Konverterer logits til sandsynligheder for hver klasse


            preds.extend(torch.argmax(probs, dim=1).cpu().numpy())      # Finder klassens indeks med højest sandsynlighed for hver input, flytter til CPU og tilføjer til preds-listen
            true.extend(labels.cpu().numpy())       # Flytter de rigtige labels til CPU og tilføjer til true-listen

    # Udskriv rapport og forvekslingsmatrix       
    print(classification_report(true, preds, target_names=['Sikker','Mistænkelig'], digits=4))     # true og preds er lister lavet oven over
    # confusion_matrix(true, preds): returnerer numpy.ndarray, shape [2,2]
    print("Confusion matrix:\n", confusion_matrix(true, preds))


# Hovedfunktion der styrer hele træningsflowet          
def main(data_path, model_name='bert-base-uncased', epochs=3, batch_size=16, lr=2e-5, max_length=128):
    # Tjek om datafil findes (string til sti)
    if not os.path.isfile(data_path):   # os.path.isfile returnerer bool
        print(f"Error: Filen '{data_path}' findes ikke.")
        sys.exit(1) # Stop program med exit-kode 1 (fejl)

    # Vælg device til beregninger: GPU (cuda) eller CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Brug GPU hvis muligt
    set_seed()  # Sæt random seed

    # Hent tokenizer og model fra HuggingFace
    tokenizer = BertTokenizerFast.from_pretrained(model_name)  # Hent tokenizer
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)  # Hent model
    # model: BertForSequenceClassification, placeret på device

    # Opret DataLoader til træning og validering
    # train, og val loader er variabler fra torch.utils.data.Dataloader
    train_loader, val_loader = create_data_loaders(data_path, tokenizer, max_length, batch_size)    #Create data loader = pytorch klasse
    
    # Opsæt optimizer og scheduler
    optimizer = AdamW(model.parameters(), lr=lr)  # Opretter en AdamW-optimizer, som opdaterer modellens vægte under træning.

    total_steps = len(train_loader) * epochs  # Antal træningssteg      # Beregner det samlede antal træningssteg (batches) i hele træningsforløbet.

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),        # int: 10% warmup
        num_training_steps=total_steps      # int: samlet antal træningssteg
    )

    # Træn modellen flere gange (epochs)            #### er i gang
    for epoch in range(1, epochs + 1):      # epoch: int fra 1 til epochs inkl. går fra 1/3
        print(f"Epoch {epoch}/{epochs}")    #placeholders i f-string 
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Loss: {loss:.4f}")      # formateret float      # Udskriver loss for denne epoch, afrundet til 4 decimaler
        eval_model(model, val_loader, device)   #deaktiverer gradientberegning med torch.no_grad()
        #Flytter batch-data til device. # eval: printer metrics

    # Gem færdig model og tokenizer     
    model.save_pretrained('model_output')   # opretter/overskriver mappe model_output med vægte og config
    tokenizer.save_pretrained('model_output')   # gemmer tokenizer-filer i samme mappe
    print("Model gemt i folderen 'model_output'")

    # Interaktiv test: Brugeren skriver selv beskeder           
    print("\nKlassificér egne beskeder (tryk Enter for at afslutte):")      #printes i k onsollen
    while True:
        txt = input("Besked: ")
        if not txt:
            break

        # metode    # Tokenizér brugerens besked, så den kan bruges som input til modellen
        enc = tokenizer.encode_plus(txt, add_special_tokens=True, max_length=max_length,
        padding='max_length', truncation=True, return_tensors='pt')     #truncation  Afkorter lange beskeder til max_length.
        inp = enc['input_ids'].to(device)       # torch.Tensor, shape [1, max_length]
        att = enc['attention_mask'].to(device)      # torch.Tensor, shape [1, max_length]

    # køres uden gradient
        with torch.no_grad():       # Deaktiverer gradientberegning (hurtigere og bruger mindre hukommelse, da vi kun skal forudsige, ikke træne)
            out = model(input_ids=inp, attention_mask=att)  # out: SequenceClassifierOutput
        probs = torch.softmax(out.logits, dim=1)[0] # torch.Tensor, # Konverterer logits til sandsynligheder for hver klasse (fx [0.8, 0.2])
        labels = ['Sikker', 'Mistænkelig']
        idx = torch.argmax(probs).item()    # idx: int# Finder index for den klasse med højest sandsynlighed (0 eller 1)
        print(f"Output: {labels[idx]} (Sikker={probs[0]:.2f}, Mistænkelig={probs[1]:.2f})\n")

# Start programmet og tillad argumenter             
if __name__ == '__main__': # Tjekker om modulet køres direkte
    import argparse     # modul til kommandolinje-argumenter

    # Opretter et ArgumentParser-objekt, som kan håndtere og forklare kommandolinje-argumenter
    parser = argparse.ArgumentParser(description='Crime-detection NLP tool')

    # Tilføjer argumenter, som brugeren kan angive ved opstart af programmet
    parser.add_argument('--data_path', type=str, default='data.csv', help='CSV med text,label')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()  #læser parametre man har skrevet i terminalen   # args er et Namespace-objekt med alle arguementer
    # Hvis brugeren ikke angiver noget, bruges standardværdierne ovenfor.

    # Kør programmet med valgte argumenter          
    main(
        args.data_path, args.model_name, args.epochs,
        args.batch_size, args.lr, args.max_length
    )





