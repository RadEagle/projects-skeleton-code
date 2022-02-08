temp = pd.read_csv(file_path, skiprows = lambda x: x in range(1, 20600), header=0)
print(len(temp))