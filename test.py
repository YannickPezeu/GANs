import json
A= json.dumps([1,2,3])
print(A)
print(type(A))
print()

B = json.loads(A)
print(B)
print(type(B))

C = json.dumps({'A': ["hello", "hello"],
                "B": "Goodbye"})
print(C)
print(type(C))

D = json.loads(C)
print(D)
print(type(D))
