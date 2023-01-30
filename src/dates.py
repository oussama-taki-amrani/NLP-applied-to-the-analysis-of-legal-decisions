# ------- Date formats --------------------

dd = "(0[1-9]| [1-9]|1[0-9]|2[0-9]|3[0-1]|1er)"
mm = "(0[1-9]|1[0-2])"
yyyy = "([1-2] {0,1}[0-9][0-9][0-9])" # allows max one space in y yyy

# date in format dd/ or dd- or dd.  with dd between 01 and 31 :
dd_f1 = dd + " *[(\/|\-|\.)]"

# month in format mm/ or mm- or mm.  with mm between 01 and 12 :
mm_f1 = " *" + mm + " *[(\/|\-|\.)]"

# year in format yyyy  with yyyy between 1000 and 2999 :
yy_f1 = " *" + yyyy

# date in the dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy format (spaces allowed between / and numbers)
date_f1 = dd_f1 + mm_f1 + yy_f1

months = "(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|decembre|fevrier|aout)"

# date in the dd month yyyy format (spaces allowed between / and numbers)
date_f2 = dd + " *" + months + " *" + yyyy


pattern_f1 = "(?:0[1-9]| [1-9]|1[0-9]|2[0-9]|3[0-1]|1er) *[(\/|\-|\.)] *(?:0[1-9]|1[0-2]) *[(\/|\-|\.)] *(?:[1-2] {0,1}[0-9][0-9][0-9])"
pattern_f2 = "(?:0[1-9]| [1-9]|1[0-9]|2[0-9]|3[0-1]|1er) *(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|decembre|fevrier|aout) *(?:[1-2] {0,1}[0-9][0-9][0-9])"


# date format :

date_f = "(" + date_f1 + "|" + date_f2 + ")"
pattern_f = r"(?:" + pattern_f1 + "|" + pattern_f2 + ")"


# -----------------------------------------