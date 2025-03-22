# 数据样例

## 预训练数据样例

每条（每行）数据的格式如下：

- **`<s>`**：表示一个句子的开始。通常在 NLP 模型中，`<s>` 被用作句子的起始标记，有助于模型理解文本的结构。

- **`<s> ... </s>`**：这对标记将文本进行分段，每段数据之间由 `<s>` 和 `</s>` 包围，通常用于在训练过程中明确分隔每个输入的句子或段落。

```json
{
  "text": "<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s>\n<s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s>\n<s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s>\n<s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s>\n<s>非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。</s>\n<s>帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。</s>\n<s>回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。</s>\n<s>识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。</s>"
}

```

## SFT 数据样例

每条（每行）样例数据样例：

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "请告诉我在中国古代的“四大发明”是什么？"
    },
    {
      "role": "assistant",
      "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：\n\n1. **造纸术**：据史书记载，东汉时期的蔡伦改进了造纸工艺，使得纸张的生产更加便捷、成本更低，质量也更加优良。这一发明极大地促进了文化的传播和保存。\n\n2. **印刷术**：中国古代的印刷术最早可以追溯到唐代的雕版印刷，到了宋代发展出了活字印刷技术。印刷术的发明极大地促进了知识的传播，降低了书籍的成本，对教育和文化的发展起到了重要的推动作用。\n\n3. **火药**：火药最初是在唐代被发现的，最初可能用于医疗或炼金术。到了宋代，人们开始将火药用于军事目的，发明了各种火器。火药的发明改变了战争的面貌，对世界军事技术的发展产生了重大影响。\n\n4. **指南针**：指南针最初在中国被用于风水测量，后来逐渐发展成为航海导航的重要工具。这一发明极大地促进了海上航行技术的发展，对于新航路的开辟和世界地理大发现起到了关键作用。\n\n这四项发明不仅在中国历史上占有重要地位，而且对全世界的科技进步和文明发展都产生了深远的影响。"
    }
  ]
}
```

## RLHF 数据样例

- "chosen" 部分： 包含正确的回答  <br>
- "rejected" 部分： 包含较差的回答

```json
{
  "chosen": [
    {
      "content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?",
      "role": "user"
    },
    {
      "content": "To determine the number of moles of HBr required to react with 2 moles of C2H6 (ethane) to form 2 moles of C2H5Br (ethyl bromide) and 2 moles of H2, we need to write the balanced chemical equation for the reaction.\n\nThe reaction is a halogenation reaction, specifically the reaction of ethane with hydrogen bromide (HBr) to form ethyl bromide and hydrogen gas. The balanced chemical equation for this reaction is:\n\n\\[ \\text{C}_2\\text{H}_6 + \\text{HBr} \\rightarrow \\text{C}_2\\text{H}_5\\text{Br} + \\text{H}_2 \\]\n\nHowever, this equation is not balanced as written. To balance it, we need to ensure that the number of atoms of each element on the reactant side is equal to the number of atoms of that element on the product side. The balanced equation is:\n\n\\[ \\text{C}_2\\text{H}_6 + \\text{HBr} \\rightarrow \\text{C}_2\\text{H}_5\\text{Br} + \\text{H}_2 \\]\n\nUpon closer inspection, we see that the equation is already balanced as written, with 1 mole of C2H6 reacting with 1 mole of HBr to produce 1 mole of C2H5Br and 1 mole of H2.\n\nGiven that you have 2 moles of C2H6, and the stoichiometry of the reaction is 1:1 for C2H6 to HBr, you would need 2 moles of HBr to react with 2 moles of C2H6 to form 2 moles of C2H5Br and 2 moles of H2.\n\nTherefore, the number of moles of HBr required is **2 moles**.",
      "role": "assistant"
    }
  ],
  "rejected": [
    {
      "content": "How many moles of HBr are required to react with 2 moles of C2H6 to form 2 moles of C2H5Br along with 2 moles of H2?",
      "role": "user"
    },
    {
      "content": "To answer this question, we need to write down the chemical equation representing the reaction between hydrogen bromide (HBr) and ethane (C2H6):\n\nC2H6(g) + 2HBr(g) \u2192 C2H5Br(l) + H2(g)\n\nNow, we can set up a mole ratio based on the coefficients in front of each substance in the balanced equation:\n\nMoles of HBr : Moles of C2H6 :: 2 : 1\n\nWe also know that there are two moles of C2H6 present in our starting mixture, so we can substitute this value into the above mole ratio to determine the amount of HBr needed:\n\n(2 mol C2H6)/(1 mol C2H6) x 2 mol HBr = 4 mol HBr\n\nSo, four moles of HBr would be required to completely react with two moles of C2H6 according to the given stoichiometric relationship.",
      "role": "assistant"
    }
  ]
}
```