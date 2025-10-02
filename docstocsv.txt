import os
import csv

def convert_docs_to_csv(input_directory, output_directory):
    # Δημιουργία του φακέλου εξόδου αν δεν υπάρχει
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Διαβάζουμε όλα τα αρχεία από το φάκελο εισόδου
    files = os.listdir(input_directory)

    for filename in files:
        input_path = os.path.join(input_directory, filename)
        output_filename = filename.split('.')[0] + '.csv'  # Ονομάζουμε το αρχείο με βάση το αρχικό όνομα
        output_path = os.path.join(output_directory, output_filename)

        # Διαβάζουμε το περιεχόμενο του αρχείου
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Γράφουμε το περιεχόμενο σε μορφή .csv
        with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Content'])  # Κεφαλίδα για το CSV αρχείο
            writer.writerow([content])    # Γράφουμε το περιεχόμενο του αρχείου

        print(f"Converted {filename} to {output_filename}")

# Διεύθυνση φακέλου εισόδου και εξόδου
input_directory = 'Collection/docs'  # Προσαρμόστε αν χρειάζεται
output_directory = 'Collection/docscsv'  # Προσαρμόστε αν χρειάζεται

# Κλήση της συνάρτησης για μετατροπή
convert_docs_to_csv(input_directory, output_directory)
