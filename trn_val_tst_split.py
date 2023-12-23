#%%
from data import parse_guitarset_filename
import glob
import pandas as pd

fps =  glob.glob("./data/GuitarSet/annotation/*.jams")

records = []
for fp in fps:
    filename = fp.split("/")[-1]
    performer_id, genre_id, progression_id, bpm, key, solo_or_comp = parse_guitarset_filename(filename)
    record = { "performer_id": performer_id, "genre_id": genre_id, "progression_id": progression_id, "bpm": bpm, "key": key, "solo_or_comp": solo_or_comp, "filename": filename}
    records.append(record)

df = pd.DataFrame(records)

print(f"length of df: {len(df)}")
# get unique values for each column
performer_ids = df.performer_id.unique()
genre_ids = df.genre_id.unique()
progression_ids = df.progression_id.unique()

bpms = df.bpm.unique()
keys = df.key.unique()
solo_or_comps = df.solo_or_comp.unique()

# print unique values
print("performer_ids: ", performer_ids)
print("genre_ids: ", genre_ids)
print("progression_ids: ", progression_ids)
print("bpms: ", bpms)
print("keys: ", keys)
print("solo_or_comps: ", solo_or_comps)

# set random seed
import random
SEED=0
random.seed(SEED)

# shuffle performer_ids, genre_ids, progression_ids
random.shuffle(performer_ids)
random.shuffle(genre_ids)
random.shuffle(progression_ids)

# read test split index
test_split_index_df = pd.read_csv("./splits/tst_split_index.csv")

# get test split index
N_TST = 9

test_df = pd.DataFrame(columns=["performer_id", "genre_id", "progression_id", "bpm", "key", "solo_or_comp", "filename"])

for i in range(N_TST):
    sample_index = test_split_index_df.iloc[i]
    performer_id_idx = sample_index.performer_id_idx
    genre_id_idx = sample_index.genre_id_idx
    progression_id_idx = sample_index.progression_id_idx

    performer_id = performer_ids[performer_id_idx]
    genre_id = genre_ids[genre_id_idx]
    progression_id = progression_ids[progression_id_idx]

    # find the corresponding record and remove it from df
    record = df.loc[(df.performer_id == performer_id) & (df.genre_id == genre_id) & (df.progression_id == progression_id)]
    df = df.drop(record.index)

    # add the record to test_df with concat
    test_df = pd.concat([test_df, record])

# rest of df is dev_df
dev_df = df

# check that test_df and dev_df are disjoint
assert len(set(test_df.filename).intersection(set(dev_df.filename))) == 0
print(len(set(test_df.filename).intersection(set(dev_df.filename))))

print(f"length of test_df: {len(test_df)}")
print(f"length of dev_df: {len(dev_df)}")

# save list of filenames to txt files
# only save the filenames and not the other columns
test_df.filename.to_csv("./splits/tst_filenames.txt", index=False, header=False)

N_VAL_RECORDINGS=18

# split dev_df into val_df and trn_df
# shuffle dev_df
dev_df = dev_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

val_df = dev_df.iloc[:N_VAL_RECORDINGS]
trn_df = dev_df.iloc[N_VAL_RECORDINGS:]

# check that trn_df and val_df are disjoint
assert len(set(trn_df.filename).intersection(set(val_df.filename))) == 0
print(len(set(trn_df.filename).intersection(set(val_df.filename))))


trn_df.filename.to_csv("./splits/trn_filenames.txt", index=False, header=False)
val_df.filename.to_csv("./splits/val_filenames.txt", index=False, header=False)


# # USE TUPLES
# # now do trn val split on dev_df
# # keep only progression_id, genre_id, performer_id
# split_df = dev_df[["performer_id", "genre_id", "progression_id"]]

# # keep only unique rows
# split_df = split_df.drop_duplicates()

# # shuffle split_df
# split_df = split_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# N_VAL = 9
# # split into trn and val
# val_df = split_df.iloc[:N_VAL]
# trn_df = split_df.iloc[N_VAL:]

# # join trn_df and val_df with dev_df to get the filenames
# trn_df = trn_df.merge(dev_df, on=["performer_id", "genre_id", "progression_id"])
# val_df = val_df.merge(dev_df, on=["performer_id", "genre_id", "progression_id"])

# # check that trn_df and val_df are disjoint
# assert len(set(trn_df.filename).intersection(set(val_df.filename))) == 0
# print(len(set(trn_df.filename).intersection(set(val_df.filename))))

# print(f"length of trn_df: {len(trn_df)}")
# print(f"length of val_df: {len(val_df)}")


# %%
