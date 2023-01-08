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
0. HMI
1. Initialization function
2. Downloading data from Yahoo finance
3. Data preparation - filtering and scaling
4. Creation of a training dataset
5. Machine learning model
6. Model training
7. Creation of a test dataset
8. Testing
9. Output in the form of a graph


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
0. HMI
1. Inicializační funce
2. Stahování dat z Yahoo finance
3. Příprava dat - filtrace a přeškálování
4. Tvorba trénovacího datasetu
5. Model strojového učení
6. Trénování modelu
7. Tvorba testovacího datasetu
8. Testování
9. Výstup ve formě grafu
