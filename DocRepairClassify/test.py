import generator
import para
import Image
g=generator.RoIGenerator()
for i in range(10):
    item=next(g)
    print(item[0][0].reshape(40,19))
    img = Image.fromarray(item[0][0].reshape(40,19), 'L')
    img.save(para.data_result_path+'/evaluate_results0/'+str(i)+'.png')
