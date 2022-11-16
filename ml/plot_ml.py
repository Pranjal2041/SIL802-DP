import matplotlib.pyplot as plt

data = '''
0.25 0.779 0.1395
0.5 0.9282999999999999 0.023
1 0.9464 0.014
2 0.9592 0.013814398285846562
5 0.965 0.0081
10 0.9687333333333333 0.0058379981348251845
50 0.973 0.0016500000000000403
100 0.9747 0.00019999999999997797
500 0.9816 0.0011331372379372459
1000 0.982 0.001204937757728613
10000 0.9842 0.002
100000 0.984425 0.000737500000000002
1000000 0.984 0.0014999999999998348
'''


data = [[float(x) for x in line.split()] for line in data.split('\n') if line.strip()]


# Draw a line plot with error region
plt.semilogx([x[0] for x in data], [x[1] for x in data], label='Differentially Privaete')
plt.fill_between([x[0] for x in data], [x[1] - x[2] for x in data], [x[1] + x[2] for x in data], alpha=0.2)
plt.semilogx([x[0] for x in data], [0.9849 for x in range(len(data))], label='No Privacy')
plt.fill_between([x[0] for x in data], [0.9849 - 0.002 for x in data], [0.9849 + 0.002 for x in data], alpha=0.2)
plt.legend()
plt.title("Differentially Private CNN Accuracy on MNIST")
# plt.title("Differentially Private BERT Accuracy on Health Data")
plt.xlabel('Epsilon of samples')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.grid(linestyle = '--')
plt.savefig('cnn_plot.png')
# plt.savefig('bert_plot.png')
plt.show()