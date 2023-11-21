# Import de fichier.
import csv

# Main
def main():
    with open('./EnglishHandwrittenCharacters/english.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(', '.join(row))


if __name__ == "__main__":
 main()