# OOP---Seminar-work

**EN**

This program is used for one-day prediction of the value of a stock and is built on the principle of machine learning.

Values entered by the user: 
1. Stock (must be supported by Yahoo finance and is entered in an abbreviation format, eg 'AAPL').
2. Start and end date of downloaded data.
3. Number of days from which the model will predict the next day's value.
4. Number of predicted days (these days are in the end of a downloaded dataset, the remaining data function as training days).
5. Number of epochs (number of learning cycles on total training data)

Point-by-point assembly of the program:
1. HMI
2. Initialization function
3. Downloading data from Yahoo finance
4. Data preparation - filtering and scaling
5. Creation of a training dataset
6. Machine learning model
7. Model training
8. Creation of a test dataset
9. Testing
10. Output in the form of a graph

********************************************************************************************************************************************************************

**CZ**

Tento program slouží k jednodenní predikci hodnoty cenného papíru a je sestaven na principu strojového učení.

Hodnoty zadávané uživatelem:
1. Cenný papír (musí být podporován Yahoo finance a je zadáván formátu zkratky, př. 'AAPL').
2. Počáteční a koncové datum stahovaných dat.
3. Počet dní, ze kterých bude model predikovat hodnotu následujícího dne.
4. Počet predikovaných dní (tyto dny jsou na koci stahovaných dat, zbylá data fungují jako tréninková).
5. Počet epoch (počet učebních cyklů na celkových tréninkových datech)

Bodové sestavení programu:
1. HMI
2. Inicializační funkce
3. Stahování dat z Yahoo finance
4. Příprava dat - filtrování a škálování
5. Vytvoření trénovací datové sady
6. Model strojového učení
7. Trénink modelu
8. Vytvoření testovacího datasetu
9. Testování
10. Výstup ve formě grafu
