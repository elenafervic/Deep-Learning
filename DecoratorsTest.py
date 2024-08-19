def Decorator(OldFunction):
    def Function():
        print("Hi")
        OldFunction()
    return Function#Why does it return the function

@Decorator#Need to define the old function after decorator
def TableFlip():
    return("SOmething")


print(TableFlip())

#So I guess instead of replacing or editing "OldFunction", you make a decorator and use that instead.
