import google.generativeai as genai
from load_creds import load_creds


class Model_AI:
    # def __init__(self):
    #     self.creds = load_creds
    #     genai.configure(credentials=self.creds)

    def  printInput(self, input):
        print("concac")

    def create_model(self, name, training_data):

        
        base_model = next(
            m for m in genai.list_models()
            if "createTunedModel" in m.supported_generation_methods
        )

        operation = genai.create_tuned_model(
            source_model = base_model.name,
            training_data=training_data,
            id = name,
            epoch_count = 100,
            batch_size = 1,
            learning_rate = 0.001,
        )

        print(operation)
        return operation
        
    def call_model(self, name):
        model = genai.get_tuned_model(f'tunedModels/{name}')
        print(model)

    def train_overdrive(self, name, input, output):
        genai.update_tuned_model(f'tunedModels/{name}', {'text_input': input,
                    'output': output,})
        


genai.configure(credentials=load_creds())
ai = Model_AI()
question = 'ai là chàng trai xấu nhất'
answer = 'không phải Trần Quốc Toàn'

# ai.create_model("testttt65","Ai đẹp trai nhất Việt Nam", "Trần Quốc Toàn","Tổng Thống Nga","Putin")
# ai.train_overdrive("test1", "ai là người xấu nhất", "không phải Trần Quốc Toàn")
# model = genai.GenerativeModel(model_name=f'tunedModels/{"test1"}')
# result = model.generate_content("Quốc Toàn có đẹp trai không")
# print(result.text)
# genai.delete_tuned_model(f'tunedModels/{"test1"}')