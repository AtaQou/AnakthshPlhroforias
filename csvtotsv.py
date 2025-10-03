import os

def convert_csv_to_tsv(csv_directory, tsv_output_file):
    with open(tsv_output_file, 'w', encoding='utf-8') as tsv_file:
        # Κεφαλίδα στο TSV αρχείο
        tsv_file.write("id\ttext\n")

        # Λήψη όλων των αριθμών από τα ονόματα των αρχείων .csv
        file_numbers = sorted([int(f.split('.')[0]) for f in os.listdir(csv_directory) if f.endswith('.csv')])

        # Βρίσκουμε το μέγιστο ID που θα πρέπει να χρησιμοποιηθεί
        max_id = max(file_numbers) if file_numbers else 0

        # Βρόχος για όλα τα IDs από 1 έως max_id
        for file_id in range(1, max_id + 1):
            filename = f"{file_id:05d}.csv"  # Ονομασία αρχείου στη μορφή 00001.csv, 00002.csv κλπ.
            csv_path = os.path.join(csv_directory, filename)

            # Αν το αρχείο υπάρχει, διαβάζουμε το περιεχόμενό του
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as csv_file:
                    lines = csv_file.readlines()
                    text = ' '.join(line.strip() for line in lines if line.strip())  # Συνδυάζουμε τις γραμμές σε μία
            else:
                # Αν το αρχείο δεν υπάρχει, χρησιμοποιούμε το "nothing"
                text = "EMPTY"

            # Γράφουμε το ID και το κείμενο στο αρχείο .tsv
            tsv_file.write(f"{file_id}\t{text}\n")

    print(f"Converted TSV file saved at: {tsv_output_file}")

if __name__ == '__main__':
    # Ο φάκελος που περιέχει τα αρχεία .csv
    csv_directory = 'Collection/docs-02'  # Αντικατάστησε με το σωστό path
    # Το αρχείο εξόδου .tsv
    tsv_output_file = 'collection.tsv'  # Αντικατάστησε με το σωστό path

    convert_csv_to_tsv(csv_directory, tsv_output_file)
