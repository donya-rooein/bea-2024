import pandas as pd
import json
from tqdm import tqdm
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login



login(token='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')

tqdm.pandas()


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        return result

    return wrapper


@torch.no_grad()
def rephrasing_text(model, tokenizer, full_prompt):
    messages = [{"role": "user", "content": full_prompt}]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    print("encodeds done")
    model_inputs = encodeds.to('cuda')
    print("model_inputs done")

    generated_ids = model.generate(model_inputs, max_new_tokens=256, do_sample=False, temperature=0.7)
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    output = predictions[0]

    return output


def extract_text_after_inst(output):
    # Extract text after [/INST]
    inst_index = output.find("[/INST]")
    if inst_index != -1:
        return output[inst_index + len("[/INST]"):].strip()
    else:
        return output


@time_it
def main():
    # Load the training data csv data into a DataFrame
    input_file_prompts = 'BEA2024/data/train_final.csv'
    df_texts = pd.read_csv(input_file_prompts)
    # df_texts = df_texts.head(5)

    checkpoint = "meta-llama/Llama-2-70b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                 cache_dir='/scratch/patent/',
                                                 torch_dtype="auto",
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,
                                              cache_dir='/scratch/patent/')
    model.eval()

    for index_problems, problem in df_texts.iterrows():
        # Prompt for repharasing
        prompt = """
        Rephrase the following text:\n{text}\n
        """

        # Text for repharasing
        text_to_analyze = problem['ItemStem_Text']

        # Insert the text into the prompt
        formatted_prompt = prompt.format(text=text_to_analyze)

        column_name = f'rephrase_ll70'  # Define column_name here

        try:
            result = rephrasing_text(model, tokenizer, formatted_prompt)
            result_after_inst = extract_text_after_inst(result)
            df_texts.at[index_problems, column_name] = result_after_inst

        except Exception as e:
            print(str(e))

            with open('error_log_data_ll70_2_temp7.txt', 'a') as error_log_file:
                error_log_file.write(str(e) + '\n')

    df_texts.to_csv('rephrased_training_2.csv', index=False)


if __name__ == "__main__":
    main()
