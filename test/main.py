from litegen import completion

res = completion(model='exaone3.5:2.4b',
                 messages="tell me about narendra modi in 5 paragrapsh simple",
                 gpu=True)

print(res.choices[0].message.content)
