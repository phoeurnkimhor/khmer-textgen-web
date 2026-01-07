from pydantic import BaseModel

class ModelInput(BaseModel):
    text: str

class ModelOutput(BaseModel):
    result: str

# Test valid input
inp = ModelInput(text="Hello KSK")
out = ModelOutput(result="Hello KSK processed")

print(inp)
print(out)

print(type(inp.text))
print(type(out.result))

