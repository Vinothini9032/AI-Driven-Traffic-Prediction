n,m,s=map(int(input().split()))
if m>n and m>s:

    print(m)
elif n>s and n>m:
    print(n)
else:
    print(s)