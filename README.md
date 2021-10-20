Distributed Associative Memory Network with Memory Refreshing Loss
===

Source code for '[Distributed Associative Memory Network with Memory Refreshing Loss](https://doi.org/10.1016/j.neunet.2021.07.030)'. <br>

<br>

Requirements
---
  * CUDA 9.0
  * CUDNN 7
  * python 2.7
  * tensorflow 1.12
  * dm-sonnet 1.34
  
```setup
pip install -r requirements.txt
```
  
<br>

Note
---

* Difference between 'DAM' and 'DAM_test'

The difference is whether a batch size is fixed or not. <br>
If you run the model with dynamic batch size, please use 'DAM_test', but this is slower than 'DAM'. <br>


* Data prepareation for Convexhull task

If you want to run the Convexhull task, then please follow below steps before running: <br>
1. Generate a directory named 'Convexhull_data' in this repository.
2. Go to [Download link](https://drive.google.com/drive/u/0/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU).
3. Download 'convex_hull_5-50_train.txt.zip', 'convex_hull_5_test.txt', and 'convex_hull_10_test.txt'.
4. Extract them and move them into 'Convexhull task' directory.



<br>


Training
---

After installing every required libraries, each task can be traind by below command. <br>


* Representation Recall Task
```shell
python train.py \
  --mode=RepresentationRecall \
  --babi=False \
  --dam=True \
  --num_memory_blocks=8 \
  --batch_size=16 \
  --learning_rate=1e-4 \
  --hidden_size=128 \
  --memory_address_size=32 \
  --memory_length_size=32 \
  --num_read_heads=1 \
  --N=8 \
  --bit_w=64 \
  --num_bit=8
  --min_length=8 \
  --max_length=16 \
  --training_iteration=20000 \
  --name=path/to/checkpoint_dir
```


* Copy Task
```shell
python train.py \
  --mode=Copy \
  --babi=False \
  --dam=True \
  --num_memory_blocks=2 \
  --p_re=0.1 \
  --batch_size=16 \
  --learning_rate=1e-4 \
  --hidden_size=128 \
  --memory_address_size=64 \
  --memory_length_size=36 \
  --num_read_heads=1 \
  --bit_w=8 \
  --min_length=8 \
  --max_length=32 \
  --training_iteration=10000 \
  --name=path/to/checkpoint_dir
```


* Associative Recall Task
```shell
python train.py \
  --mode=AssociativeRecall \
  --babi=False \
  --dam=True \
  --num_memory_blocks=2 \
  --p_re=0.1 \
  --batch_size=16 \
  --learning_rate=1e-4 \
  --hidden_size=128 \
  --memory_address_size=32 \
  --memory_length_size=36 \
  --num_read_heads=1 \
  --bit_w=8 \
  --min_length=2 \
  --max_length=8 \
  --item_bit=3 \
  --training_iteration=10000 \
  --name=path/to/checkpoint_dir
```


* Nth Farthest task
```shell
python run_nfar.py \
  --dam=True \
  --num_memory_blocks=6 \
  --p_re=0.3 \
  --batch_size=1600 \
  --learning_rate=1e-4 \
  --hidden_size=1024 \
  --memory_address_size=16 \
  --memory_length_size=128 \
  --num_read_heads=4 \
  --training_iteration=300000 \
  --name=path/to/checkpoint_dir
```


* Convexhull Task
```shell
python run_convexhull.py \
  --dam=True \
  --num_memory_blocks=6 \
  --p_re=0.3 \
  --batch_size=128 \
  --learning_rate=1e-4 \
  --hidden_size=256 \
  --memory_address_size=20 \
  --memory_length_size=64 \
  --num_read_heads=4 \
  --training_iteration=300000 \
  --name=path/to/checkpoint_dir
```





* bAbI Task
  * Training
  ```shell
    python train.py \
      --babi=True \
      --dam=True \
      --num_memory_blocks=2 \
      --p_re=0.1 \
      --batch_size=32 \
      --learning_rate=3e-5 \
      --hidden_size=256 \
      --memory_address_size=128 \
      --memory_length_size=48 \
      --num_read_heads=4 \
      --epoch=50 \
      --name=path/to/checkpoint_dir
  ```
  
  * Fine-tuning
  ```shell
    python train.py \
      --babi=True \
      --dam=True \
      --num_memory_blocks=2 \
      --p_re=0.1 \
      --batch_size=32 \
      --learning_rate=1e-5 \
      --hidden_size=256 \
      --memory_address_size=128 \
      --memory_length_size=48 \
      --num_read_heads=4 \
      --epoch=5 \
      --name=path/to/checkpoint_dir
  ```

Evaluation
---

* bAbI Task

```shell
python eval.py \
  --dam=True \
  --num_memory_blocks=2 \
  --hidden_size=256 \
  --memory_address_size=128 \
  --memory_length_size=48 \
  --num_read_heads=4 \
  --name=path/to/checkpoint_dir \
  --num=<the number of training iterations>
```


# Citation
```
@article{park2021distributed,
  title={Distributed associative memory network with memory refreshing loss},
  author={Park, Taewon and Choi, Inchul and Lee, Minho},
  journal={Neural Networks},
  volume={144},
  pages={33--48},
  year={2021},
  publisher={Elsevier}
}
```



Acknowledement
---
DNC model code based on [DeepMind's DNC](https://github.com/deepmind/dnc). <br>
Reference code for Convexhull task and Nth Farthest task [[link](https://github.com/thaihungle/SAM)].
