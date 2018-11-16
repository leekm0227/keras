from study import Exam


test = Exam()

model = test.load_model('model')
model1 = test.load_model('model1')
#Exam.train(model)
test.test(model)
print('='*30)
test.test(model1)
#Exam.save_model(model, 'model1')
