Write code to help me finish the following tasks. 

- Download the Llama 8B instuct model from hugging face.
- Deloy the model locally (assume that I have one A100)
- Test the model on the following task. Prompt should be something like (Feel free to modify the prompt): "Task: given t0, t1, t2, t3, t4, t5, output= ((t2!=t4) and (t=0!=t5)) or (t1==t3). Now t0={t0}, t1={t1}, t2={t2}, t3={t3}, t4={t4}, t5={t5}, output=?(true or false) Answer:" You should sample tokens t0-t5 from the vocabulary of the model to constuct a test set. Report the accuracy of the task. 