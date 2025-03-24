# viet-news-sum-mt5-small-finetune

## Training Notbook
  Available at: https://github.com/danhtran2mind/viet-news-sum-mt5-small-finetune/blob/main/notebooks/viet-news-sum-mt5-small-finetune.ipynb

## Dataset
  Available at: https://huggingface.co/datasets/OpenHust/vietnamese-summarization
  
## Base Model
  Available at: https://huggingface.co/google/mt5-small
  
## Training Hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-4
- train_batch_size: 12
- eval_batch_size: 12
- seed: 42
- weight_decay: 0.01
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- num_epochs: 50

## Metric
- Training loss: 0.052300
- Validation loss: 0.006372
- BLEU Score on Validation Set: 0.9964783232500736

## Dependencies Version

### Python Version
  Version: 3.10.12
  
### Libraries Version
```python
pandas==2.2.3
numpy==1.26.4
torch==2.5.1
nltk==3.2.4
pytorch-cuda==12.1
datasets==3.3.1
tqdm==4.67.1
transformers==4.47.0
```

## How to Use

### Installation & Setup
```python
pip install -r requirements.txt
```
### Import Libraries
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
```

### Load the Model
```python
model_name = "danhtran2mind/viet-news-sum-mt5-small-finetune"
tokenizer = T5Tokenizer.from_pretrained(model_name)  
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

### Inference Step
```python
def preprocess_input(text):
    inputs = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    return inputs

def generate_summary(text):
    inputs = preprocess_input(text)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128, 
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

input_text = """
Vào ngày 8-1, khoa gây mê hồi sức Bệnh viện Đa khoa Đức Giang tiếp nhận bệnh nhân L.T.N.T. (23 tuổi, Chương Mỹ, Hà Nội) trong tình trạng hôn mê sau tai nạn giao thông.

Thai phụ mang thai 26 tuần bị viêm phổi, chấn thương sọ não nghiêm trọng với xuất huyết dưới nhện và tụ máu dưới màng cứng trán phải.

Theo bác sĩ Lê Nguyễn An - trưởng khoa gây mê hồi sức Bệnh viện Đa khoa Đức Giang, vấn đề thách thức trong quá trình điều trị với bệnh nhân này là việc cần phải đảm bảo sức khỏe cho cả mẹ và con là rất khó khăn.

"Các bác sĩ cố gắng duy trì tuổi thai ngoài 30 tuần để đảm bảo việc khi sinh ra trẻ có thể phát triển bình thường. Việc đảm bảo an toàn tính mạng cho mẹ cũng phải cân đối phù hợp, hạn chế tối thiểu việc ảnh hưởng tới thai nhi", bác sĩ An nói.

Trong suốt quá trình điều trị, các bác sĩ liên tục phối hợp với chuyên khoa sản và dinh dưỡng để đánh giá và điều chỉnh liên tục cho người bệnh để đảm bảo sự phát triển của em bé trong bụng mẹ.

Đặc biệt việc chăm sóc người bệnh ở trạng thái hôn mê, thở qua mở khí quản rất khó khăn, nhiều nguy cơ rủi ro về tình trạng nhiễm khuẩn, thiếu hụt dinh dưỡng, loét trợt điểm tì đè, nguy cơ suy thai".

Sau 70 ngày điều trị, tình trạng của sản phụ dần ổn định. Các chỉ số sinh tồn cải thiện, bệnh nhân tự thở qua mở khí quản, thai phát triển bình thường.

Tối 15-3, sản phụ có dấu hiệu chuyển dạ, thai 36 tuần (theo dự kiến sinh), ngôi ngược, ối vỡ sớm. Đội ngũ bác sĩ quyết định mổ lấy thai.

Ca phẫu thuật thành công, một bé trai nặng 2kg chào đời khóc to, niêm mạc hồng hào trong niềm hạnh phúc vô bờ của đội ngũ y bác sĩ và gia đình.

Ba ngày sau mổ, sản phụ tỉnh táo, tự ăn uống, được rút mở khí quản. Dự kiến cả mẹ và bé xuất viện trong ngày 21-3.
"""

input_text = input_text.replace("\n", "")
summary = generate_summary(input_text)
print(f"Summary: {summary}")
```
# Example output
```python
# Summary: Đến khoa gây mê hồi sức Bệnh viện Đức Giang , Hà Nội vừa tiếp nhận một nữ thai phụ mang thai 26 tuần bị viêm phổi , chấn thương sọ não nghiêm trọng với xuất huyết dưới nhện .
```


