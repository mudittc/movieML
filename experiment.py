import math
A = [1, 1, 0]
B = [1, 0, 1]
x = 0
for i in range(len(A)):
    y = A[i] * B[i]
    x += y

print(x)
z = 0
for i in range(len(A)):
    w = A[i] * A[i]
    z += w
result = math.sqrt(z)
print(result)

o = 0
for i in range(len(B)):
    w = B[i] * B[i]
    o += w

result2 = math.sqrt(o)
print(result2)

similarity = x/(result2 * result)
print(similarity)