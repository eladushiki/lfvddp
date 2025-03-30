# Retrieved from /srv01/agrp/yuvalzu/scripts/terminal_scripts/copy_txt_to_csv.py @ 2025-01-26

# todo: move to a function
import glob,os,sys,csv
csv_file = sys.argv[1]
txt_files = sys.argv[2]
dir = os.getcwd()+'/'
csv_file = dir+csv_file
txt_files = dir+txt_files+'*txt'
if csv_file[-4:]!=".csv":
    csv_file += ".csv"
files = glob.glob(txt_files) 
with open(csv_file, mode='w', newline='') as csvf:
    writer = csv.writer(csvf, delimiter=',')
    for file in files:
        with open(file,'r') as txtf:
            content = float(txtf.read())
        writer.writerow([content,file])
