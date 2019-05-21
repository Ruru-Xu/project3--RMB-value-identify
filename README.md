# project3--RMB-value-identify

The aim of this project : **Identify the face value of RMB**    I got the score: 99.98

like the image:

![1558438993514](img/1558438993514.png)

I wrote the entire implementation process on my GitHub.https://github.com/xr0927/project3--RMB-value-identify

------

## my environment：

keras + tensorflow-gpu==1.10

Quadruple TITAN Xp Server

------

## step1. make datasets

- **original datasets：**train folder + label.csv + test folder

1. under the *train* folder：

![1558440064220](img/1558440064220.png)

2. under the *label.csv*:

![1558440383065](img/1558440383065.png)

3. under the *test* folder : 20,000 pictures

- **My dataset format:**

  ![1558441040945](img/1558441040945.png)

*Description* : I divided the *original training datasets* by RMB value and stored them in the corresponding folder.

(you know, under the *train* folder, the name of folders are the RMB values)------------------------>And then divide them into *train_split* of 80% and *val_split* of 20%--------------------------->And then I put the original test_data directly under the *test_RMB* folder.

**Let's do it !!!!!!!!!!!**

1. Processing *train_face_value_label.csv* files : Divided by label and stored in the corresponding  **.txt file*.

​       These operations are all done in excel(Converting formats, sorting, extracting, etc. in excel,all of these operations     are so easy)

![1558442284995](img/1558442284995.png)![1558442354740](img/1558442354740.png)

2. run *divide_cls.py* : According to the picture name in the *.txt file, and find the corresponding picture under the train folder, then move them to the corresponding label folder.

   ![1558443571829](img/1558443571829.png)

3. run *split_train_val.py* :divide train datasets into *train_split* of 80% and *val_split* of 20%

   ------

## step2. Training

run *train.py*

![1558449440469](img/1558449440469.png)

![1558444060674](img/1558444060674.png)

I find that *val_acc did not improve from 0.99922*, so I kill it

------

## step3. testing

1. run *test_averageCSV.py* : Generate *.csv* file

   ![1558454997990](img/1558454997990.png)
   
   ![1558449486650](img/1558449486650.png)

![1558444546942](img/1558444546942.png)

you must be know what it means I guess, I don't have to explain it anymore.

2. run *last_max_valueCSV.py* :Extract the maximum value of each row in the above table in order to generate the final commit .csv file

![1558444994019](img/1558444994019.png)

![1558445170325](img/1558445170325.png)

