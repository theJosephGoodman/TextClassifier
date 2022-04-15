# Анализатор текста

## Описание:
В корневой папке находится папка solution - в нём есть файл solution_kontur.ipynb в котором можно проследить за ходом моей работы.
Файл program.py - готовый файл, который выполняет работу по классификации текста. Для его работы необходимо наличие в  каталоге solution папки с моделью rubert, файла логистической регрессии от sklearn и текстового файла в котором хранится классифицируемый текст

ссылка на модель rubert: https://huggingface.co/DeepPavlov/rubert-base-cased-sentence


## Цель:
На основе текста заголовка новости даёт прогноз о её правдивости


## Используемые библиотеки:
numpy, torch, pandas, matplotlib, pymorphy2, sklearn, nltk, pickle


## Логика работы:
Препроцессинг + эмбединги на основе модели rubert + фича длины предложения -> логистическая регрессия


