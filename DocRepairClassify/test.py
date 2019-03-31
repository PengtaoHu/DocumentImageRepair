import generator
train_generator = generator.DataGenerator()
t=next(train_generator)
print(t[0].shape,t[1].shape)
val_generator = generator.ValidationDataGenerator()
t=next(val_generator)
print(t[0].shape,t[1].shape)
