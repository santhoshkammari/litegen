from litegen import completion, pp_completion

pp_completion(model='exaone3.5:2.4b',
              messages="tell me about narendra modi in 5 paragrapsh simple",
              gpu=True)
