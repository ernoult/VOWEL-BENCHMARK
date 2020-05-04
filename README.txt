#to visualize any database (for instance R = 10):
python main.py --action data --R 10

#to run a training simulation
python main.py --action train --epochs 100 --lr 0.05

#to get the csv files of the results for all the vowels, all the noise level, 10 times each
python main.py --action results
