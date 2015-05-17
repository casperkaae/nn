local HardSigmoid = torch.class('nn.HardSigmoid', 'nn.Module')

function HardSigmoid:updateOutput(input)
   return input.nn.HardSigmoid_updateOutput(self, input)
end

function HardSigmoid:updateGradInput(input, gradOutput)
   return input.nn.HardSigmoid_updateGradInput(self, input, gradOutput)
end
