# 理论基础

- 回溯三问
  - 操作问题？
  - 子问题？
  - 下一个子问题？

## 从记忆化搜索到递推（把 dfs 改成一个数组，递归改成循环打破）

### 198.打家劫舍

> 中等

降低问题规模：从第一个/最后一个房子开始选

- 三问
  - 当前房子选不选？
  - 从 **前** i 个房子中得到的最大金额和
  - 分类讨论
    - 不选：前 i-1 个房子中得到的最大金额和
    - 选：前 i-2 个房子中得到的最大金额和
  - 写出转移方程
- 我的思路
  - 用一个 dp 数组存 **前** i 个房子的最大金额和
    - 长度比房屋数组多 1——第 0 个元素存 0
      - **第** 2 个房子计算 **前** 2 个房子的最大金额时时不会越界
      - dp 数组下标意义：**前** i 个房子的最大金额和

### 基本思路

- 回溯递归

  - 超时

- 记忆化搜索
  还是递归

  - 用一个全局变量数组——存储已经计算过一次的和
  - 时/空：O(n)

- 递推
  空：O(1)

  - O(n)：我的思路，

  - O(1)：但是其实 **不用 dp 数组存**

    - 我的思路：直接修改原数组

    - **要处理 i 为 0 和 1 的情况**

      - ```python
        if nums[0] > nums[1]:
            nums[1] = nums[0]
        ```

  - 优化的思路

    - **用两个变量** 分别记住前两个状态的值
      - 代码会更简洁
      - 不需要处理 i 为 0 和 1 的情况

# 0-1 背包

==python 装饰器：`@cache`==

### 三种常见变形

- 至多装 capacity，求方案数/最大价值和
- 恰好装 capacity，求方案数/最大/最小价值和
- 至少装 capacity，求方案数/最小价值和

### 494.目标和（常见的变形）

> 中等
>
> 可复习

- 三问
  - 选不选当前物品
  - 剩余容量为 c 时，前 i 个物品中得到的最大价值和
  - 分类讨论
    - 不选：剩余容量为 c 时，前 i 个物品中得到的最大价值和
    - 选：剩余容量为 c-w [i] 时，前 i 个物品中得到的最大价值和

- 本题并不能直接套用 0-1 背包
  - 需要进行转换

#### ==转换思路==：将决定每个元素的符号转换为——找到和为 p 的 所有符号为正的数字和的方案

本质：s 和 target 都为 **已知** 条件

- 假设 **所有元素（此时没有负号）** 和为 s，所有 **符号为正** 的数字和为 $p$，则所有 **符号为负** 的数字和为 $-(s-p)$
  - $target = p-(s-p)$
    - $p={(s+target)}/2$
    - **隐含**：$(s+target)$ 必须为 **非负偶数**
  - 这样就转换为了 **找到数的和为 p 的方案数**（剩余数为负数）

- 恰好----方案数

### 优化的回溯代码：==零一背包的封装代码==（选/不选当前值）

注意区别最值问题和方案数问题

```python
def zero_one_knapsack(capacity: int, w: List[int], v: List[int]) -> int:
    n = len(w)
    
    @cache
    def dfs(i, c):
        if i < 0:
            return 0
        # 该元素超出容量
        if c < w[i]:
            return dfs(i-1, c)
        ## 求最大、最小价值
        return max(dfs(i-1, c), dfs(i-1, c-w[i]) + v[i])
    	## 求方案数（两种情况，选还是不选）
        ## return dfs(i-1, c) + dfs(i-1, c-w[i])
    
    return dfs(n-1, capacity)
```

### 记忆化搜索

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        n = len(nums)
        target += sum(nums)
        if target < 0 or target % 2:
            return 0
        target //= 2
		
        @cache
        def dfs(i, c):
            if i < 0:
                return 1 if c==0 else 0
            if c < nums[i]:
                return dfs(i-1, c)
            return dfs(i-1, c) + dfs(i-1, c-nums[i])
        
        return dfs(n-1, target)       
```

时间复杂度：状态的个数($n*target$)乘以每个状态所需的时间 $O(1)$

空间复杂度：同时间

### 递推

- 用数组和循环替代递归

- 数组的设置：**dp\[i\]\[j\]：前 i 个数字能凑成和为 j 的放番薯**

  - 每个元素最多有 target 种状态
    避免出现负数下标

    - `dp = [[0] * (target+1) for _ in range(n+1)]`
      - `dp[0][0] = 1`——初始化
        - **此时只有一种情况可以得到和 0，那就是“不选任何数字”**
        - 因此初始化为 1

  - 对于每种状态

    - ```python
      for i, x in enumerate(nums):
      	for c in range(target+1):
      ```

code

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # p s-p
        # p = (s+target)/2
        n = len(nums)
        target += sum(nums)
        if target < 0 or target % 2:
            return 0
        target //= 2
		# 数组设置
        dp = [[0] * (target+1) for _ in range(n+1)]
        dp[0][0] = 1

        for i, x in enumerate(nums):
            for c in range(target+1):
                if c < x:
                    dp[i+1][c] = dp[i][c]
                else:
                    dp[i+1][c] = dp[i][c] + dp[i][c-x]
        return dp[n][target]
```

#### 优化 1：只有两个数组(0-1)

- 计算完 dp [i+1] 后，就不再需要 dp [i] 了
  - 每时每刻只有两个数组中的元素在转移
    - 将原来的数组压缩为 **两行**，交替使用
    - 对比打家劫舍：将数组压缩为 **两个变量**

```python
# 投影到0/1中——0/1背包
dp = [[0] * (target+1) for _ in range(2)]
dp[0][0] = 1

for i, x in enumerate(nums):
    for c in range(target+1):
        if c < x:
            dp[(i+1)%2][c] = dp[(i)%2][c]
        else:
            dp[(i+1)%2][c] = dp[(i)%2][c] + dp[(i)%2][c-x]
return dp[n%2][target]
```

空间复杂度：O(target)

#### 优化 2：一个数组

==在使用前面的值运算时，从后往前算可以解决很多覆盖问题==——确保 **每个** 物品只能 **被选一次**

- 如果压缩为一个数组

  - 从左到右算会把正确结果覆盖掉

    - 因此：**从右到左（从后往前）算**

      - 注意遍历时的开闭端点

      - 错误的写法：

        ```python
        for c in range(target, -1):
        # 倒序时没有默认的方法！必须要指定起始点
        ```

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # p s-p
        # p = (s+target)/2
        n = len(nums)
        target += sum(nums)
        if target < 0 or target % 2:
            return 0
        target //= 2

        dp = [0] * (target+1)
        dp[0] = 1
		# 处理后
        for x in nums:
            for c in range(target, x-1,-1):
                # if c < x:
                #     break
                # else:
                dp[c] = dp[c] + dp[c-x]
        return dp[target]
    	# 降维后原来的部分
        for i, x in enumerate(nums):
            for c in range(target+1):
                if c < x:
                    dp[c] = dp[c]
                else:
                    dp[c] = dp[c] + dp[c-x]
        # 处理1：for x in nums:    
        # 处理2：if c < x，是多余的，只需要讨论 c>=x 的情况：for c in range(target, x-1,-1):
```





# 完全背包

- 与 0-1 背包区别：
  - 定义：**n 个** 物品变为 **n 种** 物品，且每种物品可以重复选
  - 回溯的区别：如果 **选** 当前 i 的物品时，递归时 **进入 i 而不是 i-1**

### 322.零钱兑换

> 中等

#### 记忆化搜索

超时

[3,7,405,436]——8839

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount==0:
            return 0
        n = len(coins)
		@cache
        def dfs(i, c):
            if i < 0:
                return 0 if c==0 else inf
            if c < coins[i]:
                return dfs(i-1, c)
            # 前一种视为：选了i x次后不再选i了
            return min(dfs(i-1, c), dfs(i, c-coins[i]) + 1)
        
        ans = dfs(n-1, amount)
        if ans < float(inf):
            return ans
        return -1
```

#### 递推

- 注意点
  - c 代表剩余容量，所以存在 `c==amount`

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount==0:
            return 0
        n = len(coins)

        dp = [[inf]*(amount+1) for _ in range(n+1)]
        # 注意是初始化为0
        dp[0][0] = 0

        for i, x in enumerate(coins):
            ### c代表剩余容量
            for c in range(amount+1):
                if c<x:
                    dp[i+1][c] = dp[i][c]
                else:
                    dp[i+1][c] = min(dp[i][c], dp[i+1][c-x] + 1)
        ans = dp[n][amount]
        return ans if ans < inf else -1
```

- 两个数组

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount==0:
            return 0
        n = len(coins)

        dp = [[inf]*(amount+1) for _ in range(2)]
        dp[0][0] = 0

        for i, x in enumerate(coins):
            for c in range(amount+1):
                if c<x:
                    dp[(i+1)%2][c] = dp[i%2][c]
                else:
                    dp[(i+1)%2][c] = min(dp[i%2][c], dp[(i+1)%2][c-x] + 1)
        ans = dp[n%2][amount]
        return ans if ans < inf else -1
```

- 一个数组

==不能和 0-1 背包一样逆序计算==——结果会错误

==此时正序计算是对的==——从前面转过来的也是 `f[i+1]` 的值，不会覆盖掉正确的答案（因为可以 **重复拿同一 ==种== 的硬币**）

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount==0:
            return 0
        n = len(coins)

        dp = [inf]*(amount+1)
        dp[0] = 0

        for x in coins:
            # 只需注意循环
            for c in range(x, amount+1):  
                dp[c] = min(dp[c], dp[c - x] + 1)
        ans = dp[amount]
        return ans if ans < inf else -1
```

# 线性 DP

- 启发思路：**子序列** 本质：**选或不选**
- 考虑最后 **一对** 字母 x、y
  - 4 种选的情况
- 三问
  - s [i]、t [j] 选不选
  - s 的前 i 个字母和 t 的前 j 个字母的 LCS 长度
  - 下一个子问题
    - s 的前 i-1 个字母和 t 的前 j-1 个字母的 LCS 长度
      - **i-1、j-1 表示不选 s [i]、t [j]**
    - s 的前 i 个字母和 t 的前 j-1 个字母的 LCS 长度
    - s 的前 i-1 个字母和 t 的前 j 个字母的 LCS 长度

### 1143. 最长公共子序列

> 中等

- **记忆化搜索（递归）** 时一定要添加 ==`@cache`==
  - 这样就不会超时了

```python
class Solution:
    def longestCommonSubsequence(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)


        @cache
        def dfs(i, j):
            if i < 0 or j < 0:
                return 0
            if s[i] == t[j]:
                return dfs(i-1, j-1) + 1
            else:
                return max(dfs(i-1, j), dfs(i, j-1))
        
        return dfs(n-1, m-1)
```

- 递推
  - 不需要初始化 dp\[0\]\[0\]
    - 因为有相等字符时才会 **加一**
    - 回溯的下界执行
      - `return 0`

```python
class Solution:
    def longestCommonSubsequence(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)

        dp = [[0] * (m+1) for _ in range(n+1)]
        ### 这里不需要初始化DP[0][0]

        for i, x in enumerate(s):
            for j, y in enumerate(t):
                if x == y:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        
        return dp[n][m]
```

- 两个一维数组
- ==一个数组==
  - 顺序遍历主要难点：防止后面的运算结果覆盖了前面的
    - ###

```python
class Solution:
    def longestCommonSubsequence(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)

        dp = [0] * (m+1)
        

        for i, x in enumerate(s):
            ### 防止前一个状态被覆盖
            pre = dp[0]
            for j, y in enumerate(t):
                ###
                tmp = dp[j+1]
                if x == y:
                    ###
                    dp[j+1] = pre + 1
                else:
                    dp[j+1] = max(dp[j+1], dp[j])
                ###
                pre = tmp
        
        return dp[m]

```

### 72. ==编辑距离==

> 困难变中等了

- 等价转换
  - 插入：在 s [i] 中插入等价于——删除 t [j] 的相同字母
  - 删除：去掉 s\[i\]
  - 位置上的字母相等：同时去掉 s\[i\]和 t [j]——**0 次操作**
  - 替换：使得字母相等，但是是 1 次操作
- 将两个字符串在以上的规则下，用最少次数变为空



- 记忆化

```python
class Solution:
    def minDistance(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)
        
    	@cache
        def dfs(i, j):
            if i < 0:
                return j + 1
            if j < 0:
                return i + 1
            if s[i] == t[j]:
                return dfs(i-1, j-1)
            # return min(dfs(i-1, j) + 1, dfs(i, j-1) + 1, dfs(i-1, j-1) + 1)
	        return min(dfs(i-1, j), dfs(i, j-1), dfs(i-1, j-1)) + 1
        return dfs(n-1, m-1)
```

- 递推
  - ==难点：翻译边界条件==
    - 错误 1：只设置了一部分
    - 错误 2：**设置列值** 的代码错误
      - 进入 **该行的循环** 时才会用到 **下一行的第 0 列** 元素
      - 在设置行值时已设置 `dp[0][0] = 0`

```python
class Solution:
    def minDistance(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)

        dp = [[inf] * (m+1) for _ in range(n+1)]
        # 正确的边界1
        dp[0] = list(range(m+1))

        # 错误2：
        # dp[:0] = list(range(n+1))
		# 错误1：
        # dp[0][1] = 1
        # dp[1][0] = 1
        # dp[0][0] = 0

        for i, x in enumerate(s):
            # 正确的边界2
            dp[i+1][0] = i+1
            for j, y in enumerate(t):
                if x == y:
                    dp[i+1][j+1] = dp[i][j]
                else:
                    dp[i+1][j+1] = min(dp[i][j+1] + 1, dp[i][j] + 1, dp[i+1][j] + 1)
        
        return dp[n][m]
```

- 一个数组
  - 难点
    - pre 的设定
      - 原来：`dp[i+1][0] = i+1` 设定的是 **下一行** 的初始值
      - pre 需要记录 **当前行** 的初始值
      - 所以：先记录 pre 再更新 `dp[0]`
    - 最小值的分支
      - 原来：`dp[i+1][j+1] = min(dp[i][j+1], dp[i][j], dp[i+1][j]) + 1`
      - 去掉第一维度后：`dp[j+1] = min(dp[j+1], dp[j], dp[j]) + 1 `
        - 后面两个值分别代表：**替换、不选** `s[i]`
        - **替换** 对应的值与 `x==y` 时应相同，都为 `pre`

```python
class Solution:
    def minDistance(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)

        if n==0 or m==0:
            return n if m==0 else m

        # dp = [inf] * (m+1)
        dp = list(range(m+1))

        for i, x in enumerate(s):
            pre = dp[0]
            dp[0] = i + 1
            for j, y in enumerate(t):
                tmp = dp[j+1]
                if x == y:
                    dp[j+1] = pre
                else:
                    dp[j+1] = min(dp[j+1], dp[j], pre) + 1
                pre = tmp
        return dp[m]
```



### 300. ==最长递增子序列==

> 中等

严格递增

#### 动态规划

- 递归
  - 我的代码：**选不选** 当前元素——需要知道上一个选的数字
    - 超出内存限制
      `1 <= nums.length <= 2500`

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)

        @cache
        def dfs(i, c):
            if i >= n:
                return 0
            if nums[i] <= c:
                return dfs(i+1, c)
            return max(dfs(i+1, c), dfs(i+1, nums[i]) + 1)
        
        return dfs(0, -inf)
```

- 这个代码似乎比较难 1:1 翻译成递推
  - 与背包不同的是，所选的值不能代表容量

- **枚举选哪个**——比较当前选的数字和下一个要选的数字
  - **只需要** 知道当前所选数字的 **下标**
  - 三问
    - 枚举 `nums[j]`
    - 子问题：以 `nums[i]` 结尾的 LIS 长度
    - 下一个子问题：以 `nums[j]` 结尾的 LIS 长度

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)

        @cache
        def dfs(i):
            res = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    res = max(res, dfs(j))
            return res + 1
        
        ans = 0
        # 枚举
        for i in range(n):
            ans = max(ans, dfs(i))
        return ans
```

- 递推（1:1 翻译）

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
		
        dp = [0] * n

        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    # 原来的res
                    dp[i] = max(dp[i], dp[j])
            # 算上自身
            dp[i] += 1

        return max(dp)
```



- 思路 3：等价于
  - nums 与 **排序去重后** 的 nums 的最长公共子序列（LCS）



#### ==贪心+二分==

- 动态规划优化时间复杂度的 ==进阶技巧==

  - 交换 **状态与状态值** 
  - `g[i]` 表示长度为 `i+1` 的 IS 的 **末尾元素最小值**
    - `len(g[i])` 表示 **所有结果中最大的长度**
    - 两种操作（顺序遍历）：
      - 遇到比末尾更大的值，则 **加入** 到数组中（后插）
      - 遇到比末尾 **更小** 的值，则 **替换** 数组中对应位置的值
        - 对应位置：
          - **替换位置 i** 的元素值 **大于或等于** 当前元素值
          - 该位置 i 之前的 `g[0:i]` 的值都比 **当前** 的元素要小
  - 变为了贪心算法
    - 没有重叠子问题

- 二分查找——==`bisect_left(list, int)`==
  **返回值范围** 为：`0~len(g)`
  返回值的意义：g [j] 的元素大于或等于 x（二分法下界）——上界 `right`

  ```python
  from bisect import bisect_left
  
  g = []
  x = 1
  # STL 库函数
  j = bissect_left(g, x)
  ```

  - 查找 **替换位置**

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        g = []

        for x in nums:
            # 二分法的下界
            j = bisect_left(g, x)
            if j == len(g):
                g.append(x)
            else:
                g[j] = x
        
        return len(g)
```

- 时间复杂度 `O(nlogn)`，空间 `O(n)`

- 空间 `O(1)`：直接对 `nums` 数组动手
  - 需要设置变量记录不存在的 g 的长度

- 如果是非严格递增

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        g = []

        for x in nums:
            # 找二分法的上界
            j = bisect_right(g, x)
            if j == len(g):
                g.append(x)
            else:
                g[j] = x
        
        return len(g)
```

# 状态机 DP

> 买卖股票

- 从最后一天开始（从第0天到第n天的最大利润）
- 三问
  - 启发：最后一天的行为？
  - 子问题：第i天结束时的利润+第5天的利润（三种情况）
  - 下一个子问题：第i-1天结束时，持有/未持有股票的最大利润
- 定义
  - `dfs(i, 0)`：表示第i天**结束**时，未持有股票的最大利润
  - `dfs(i, 1)`：表示第i天**结束**时，持有股票的最大利润
- 递归边界
  用负值会比较不麻烦（prices数组的空值问题）
  - `dfs(-1, 0)=0`
    - 第0天开始未持有股票，利润为0
  - `dfs(-1,1)=-inf`
    - 第0天开始不可能持有股票
- 递归入口：
  - `dfs(n-1, 0)`

## 122. 买卖股票的最佳时机II

> 中等

### 记忆化搜索

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, j):
            if i < 0:
                return 0 if j == 0 else -inf
            if j == 0:
                return max(dfs(i-1, 1) + prices[i], dfs(i-1, 0))
            else:
                return max(dfs(i-1, 1), dfs(i-1, 0) - prices[i])
        
        return dfs(n-1, 0)
```

### 递推

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        
        dp = [[0] * 2 for _ in range(n+1)]
        dp[0][0] = 0
        dp[0][1] = -inf

        for i, price in enumerate(prices):
            dp[i+1][0] = max(dp[i][1] + price, dp[i][0])
            dp[i+1][1] = max(dp[i][1], dp[i][0] - price)
        return dp[n][0]
```

- 一个数组（==两个变量）

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp1 = 0
        dp2 = -inf
        for price in prices:
            dp1, dp2 = max(dp2 + price, dp1), max(dp2, dp1 - price)
        return dp1
```

### 309. 交易冷冻期

> 中等

#### 记忆化

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, j):
            if i < 0:
                return 0 if j == 0 else -inf
            if j == 0:
                return max(dfs(i-1, 1) + prices[i], dfs(i-1, 0))
            else:
                return max(dfs(i-1, 1), dfs(i-2, 0) - prices[i]) # 只有这个
        
        return dfs(n-1, 0)
```

#### 递推

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        
        dp = [[0] * 2 for _ in range(n+2)] # 注意是n+2
        dp[1][1] = -inf

        for i, price in enumerate(prices):
            dp[i+2][0] = max(dp[i+1][1] + price, dp[i+1][0])
            dp[i+2][1] = max(dp[i+1][1], dp[i][0] - price)
        return dp[n+1][0] # dp[-1][0]
```

- 空间优化

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        pre_dp1 = 0
        dp1 = 0
        dp2 = -inf
        for price in prices:
            pre_dp1, dp1, dp2 = dp1, max(dp2 + price, dp1), max(dp2, pre_dp1 - price)
            
        return dp1
```

## 188. 买卖股票的最佳时机IV——至多k次交易（买k次，卖k次）

> 困难

- 只需要在买入或卖出时才需要计数
  - k=0时合法，不发生任何交易就不会有问题

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i, hold, j):
            if j < 0:# k可以等于0，是合法的
                return -inf
            if i < 0:
                return -inf if hold else 0
            if hold:
                return max(dfs(i-1, True, j), dfs(i-1, False, j) - prices[i])
            return max(dfs(i-1, True, j-1) + prices[i], dfs(i-1, False, j))
        return dfs(n-1, False, k)
```

#### 递推

- ==三维==数组（每一维存什么？）
  - prices数组长度
  - （至多）交易的次数
  - 状态数（2）
- 初始化边界
  - 0和-inf
  - 注意k的合法范围为`(0,k)`

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        dp = [[[-inf] * 2 for _ in range(k+2)] for _ in range(n+1)] # k+2
        for j in range(1, k+2):
            dp[0][j][0] = 0 # j=0时对应k=-1
        for i, price in enumerate(prices):
            for j in range(1, k+2):
                dp[i+1][j][0] = max(dp[i][j-1][1] + price, dp[i][j][0])
                dp[i+1][j][1] = max(dp[i][j][1], dp[i][j][0] - price)
        
        return dp[n][-1][0]
```

#### 空间优化

- 顺序（python语法特性， 不需要中间变量）

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        dp = [[-inf] * 2 for _ in range(k+2)]
        for j in range(1, k+2):
            dp[j][0] = 0
        for i, price in enumerate(prices):
            for j in range(1, k+2):
                dp[j][0], dp[j][1] = max(dp[j-1][1] + price, dp[j][0]), max(dp[j][1], dp[j][0] - price)
        return dp[-1][0]
```

- 倒序遍历
  - k=0不需要

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        dp = [[-inf] * 2 for _ in range(k+2)]
        for j in range(1, k+2):
            dp[j][0] = 0
        for i, price in enumerate(prices):
            for j in range(k+1, 0, -1):# 边界
                # 先计算1，再计算0（0不需要1）
                dp[j][1] = max(dp[j][1], dp[j][0] - price)
                dp[j][0] = max(dp[j-1][1] + price, dp[j][0])
        return dp[-1][0]

```

## 恰好K次/至少K次

区别：边界条件

### 恰好k次

- 递归到`i<0`时，只有j=0合法，j>0都非法

```python
# 恰好
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        # 递推
        n = len(prices)
        f = [[[-inf] * 2 for _ in range(k + 2)] for _ in range(n + 1)]
        f[0][1][0] = 0  # 只需改这里
        for i, p in enumerate(prices):
            for j in range(1, k + 2):
                f[i + 1][j][0] = max(f[i][j][0], f[i][j][1] + p)
                f[i + 1][j][1] = max(f[i][j][1], f[i][j - 1][0] - p)
        return f[-1][-1][0]

        # 记忆化搜索
        # @cache
        # def dfs(i: int, j: int, hold: bool) -> int:
        #     if j < 0:
        #         return -inf
        #     if i < 0:
        #         return -inf if hold or j > 0 else 0
        #     if hold:
        #         return max(dfs(i - 1, j, True), dfs(i - 1, j - 1, False) - prices[i])
        #     return max(dfs(i - 1, j, False), dfs(i - 1, j, True) + prices[i])
        # return dfs(n - 1, k, False)
```

### 至少k次

- 递归到至少0次时，等价于**交易次数没有限制**，与122.的状态转移相同
  - 注意递推的写法
    - 分开无限次（j=0时蕴含<0）和k次

```python
# 至少
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        # 递推
        n = len(prices)
        f = [[[-inf] * 2 for _ in range(k + 1)] for _ in range(n + 1)]
        f[0][0][0] = 0
        for i, p in enumerate(prices):
            f[i + 1][0][0] = max(f[i][0][0], f[i][0][1] + p)
            f[i + 1][0][1] = max(f[i][0][1], f[i][0][0] - p)  # 无限次
            for j in range(1, k + 1):
                f[i + 1][j][0] = max(f[i][j][0], f[i][j][1] + p)
                f[i + 1][j][1] = max(f[i][j][1], f[i][j - 1][0] - p)
        return f[-1][-1][0]

        # 记忆化搜索
        # @cache
        # def dfs(i: int, j: int, hold: bool) -> int:
        #     if i < 0:
        #         return -inf if hold or j > 0 else 0
        #     if hold:
        #         return max(dfs(i - 1, j, True), dfs(i - 1, j - 1, False) - prices[i])
        #     return max(dfs(i - 1, j, False), dfs(i - 1, j, True) + prices[i])
        # return dfs(n - 1, k, False)
```

## 区间DP

- 与线性DP的区别：
  - 区间DP：一般在前缀、后缀上转移
  - 线性DP：从小区间转换到大区间
- 选或不选
  - 从两侧向内缩小规模
  - 516.最长回文子序列
- 枚举选哪个
  - 分割成多个规模更小的子问题
  - 1039.多边形三角剖分的最低得分

### 516.最长回文子序列

> 中等

#### 记忆化搜索

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        @cache
        def dfs(i, j):
            if i > j:
                return 0
            if i == j:
                return 1
            if s[i] == s[j]:
                return dfs(i+1, j-1) + 2
            return max(dfs(i+1, j), dfs(i, j-1))
        
        return dfs(0, n-1)
```

#### 递推

- `i`：倒序枚举
  - 从i+1转移到i
- `j`：正序枚举
- 边界条件处理
  - `i==j`时dp元素值为1
    - 这个时候dp[i\]\[j]不能直接转移——越界
    - 直接初始化对应的值，并且循环时跳过这种情况

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        
        dp = [[0] * n for _ in range(n)]
        for i in range(n-1, -1, -1):
            dp[i][i] = 1
            for j in range(i+1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][-1]
```

#### 空间优化

- 因为j必须正序遍历，所以要做好防覆盖
  - 每行开始初始化pre为0
  - 遍历每个元素时用tmp记下当前每个元素的值
  - 计算相等的情况时用pre值更新dp值
  - 每次遍历结束时更新pre的值为tmp

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        
        dp = [0] * n
        for i in range(n-1, -1, -1):
            dp[i] = 1
            pre = 0
            for j in range(i+1, n):
                tmp = dp[j]
                if s[i] == s[j]:
                    dp[j] = pre + 2
                else:
                    dp[j] = max(dp[j], dp[j-1])
                pre = tmp
        return dp[-1]
```

### 1039.多边形三角剖分的最低得分

> 中等

- 当前所取的三角形
  - 会分割出两个区域（区域面积可以为0）
- 在剩余的两个区域中计算三角形面积
- 在下一个剩余的两个区域中计算三角形面积

#### 记忆化搜索

- **边界**：i、j之间没有点，无法构成三角形
- 入口：i=0，j=n-1

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        n = len(values)
        @cache
        def dfs(i, j):
            if i+1 == j:
                return 0
            res = inf
            for k in range(i+1, j):
                res = min(res, dfs(i,k) + dfs(k, j) + values[i]*values[j]*values[k])
            return res
        return dfs(0, n-1)
```

#### 递推

- i倒序枚举
- j正序枚举

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        n = len(values)
        dp = [[0] * n for _ in range(n)]
        # for i in range(n-1):
        #     dp[i][i+1] = 0
        
        for i in range(n-3, -1, -1):
            for j in range(i+2, n):
                res = inf
                for k in range(i+1, j):
                    res = min(res, dp[i][k] + dp[k][j] + values[i]*values[j]*values[k])
                dp[i][j] = res
        return dp[0][n-1]
```

## 树形DP

### 543.二叉树的直径（类型1）

> 简单
>
> 可能不经过根结点

- 计算节点左右子树的最大长度

- 边界

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0

        def dfs(node):
            nonlocal ans
            if not node:
                # 左右子树为空，叶子节点的左右深度为0，不会+1
                return -1
            left_depth = dfs(node.left) + 1
            right_depth = dfs(node.right) + 1
            ans = max(ans, left_depth + right_depth)
            return max(left_depth, right_depth)
        dfs(root)
        return ans
```

#### 124.二叉树中的最大路径和

> 困难

- 递归过程——相当于更新节点值（代表当前节点的最大链长）
- ans 为 0 —— 相当于不选任何节点
  - 所以更新节点值时需要与0比较

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = -inf

        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            nonlocal ans
            ans = max(ans, left + right + node.val)
            ###
            return max(max(left, right) + node.val, 0)
        
        dfs(root)
        return ans
```

#### 2246.相邻字符不同的最长路径

> 困难
>
> 普通树

- 思路1：遍历x的子树，把最长链的长度都存到一个列表中，排序，取最大的两个
- 思路2：遍历**节点x**的子树同时求**对于节点x的最长+次长**
  - 如何一次遍历找到最长+次长？
    - 次长在前面，最长在后面
      - 遍历到最长的时候就能算出最长+次长
    - 最长在前面，次长在后面
      - 遍历到次长的时候就能算出最长+次长
    - **遍历时维护最长长度**，一定会在遍历到某棵子树时算出最长+次长

- 实现
  - 先记录每个节点的子结点（因为题目只提供了节点的父节点）
  - 递归
    - 隐含边界：叶结点会返回0
    - x_len：记录最大长度
    - y_len：判断是否为新的最长（直接比较）、次长（通过和的值判断）
    - 记得判断字符是否相等
      - 相等则不会更新x_len、y_len

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        g = [[] for _ in range(n)]
        # 存储邻居：子节点（不包含父节点）
        for i in range(1, n):
            g[parent[i]].append(i)

        ans = 0
        def dfs(x):
            nonlocal ans
            x_len = 0
            for y in g[x]:
                y_len = dfs(y) + 1
                if s[y] != s[x]:
                    ans = max(ans, x_len + y_len)
                    x_len = max(x_len, y_len)
            return x_len
        dfs(0)
        return ans + 1
```

### 337.打家劫舍 III（类型2）

> 中等

- 选/不选
  - 每个儿子分别存储选/不选当前节点的最高金额
    - **降低复杂度——不用遍历儿子的儿子（涉及4个节点）和判断其是否为空节点**
- 提炼状态
  - 选/不选当前节点时，以当前节点为根的**子树**最大点权和
- 转移方程
  - 选 = 左不选 + 右不选 + 当前节点值
  - 不选 = max(左选，左不选) + max(右选，右不选)
- 最终答案：max(根选，根不选)

```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        
        def dfs(node):
            if not node:
                return 0, 0
            left_s, left_ns = dfs(node.left)
            right_s, right_ns = dfs(node.right)

            cur_s = left_ns + right_ns + node.val
            cur_ns = max(left_ns, left_s) + max(right_ns, right_s)
            return cur_s, cur_ns
        
        s, ns = dfs(root)
        return max(s, ns)
```

#### 总结

- 树上的最大独立集
  - 二叉树
    - 337
  - 一般树
    - 没有上司的舞会
  - **最大独立集**需要从图中选择尽量多的点，使得这些点**互不相临**
    - 变形：最大化点权之和



### 968.监控二叉树（类型3）

> 困难

- 选/不选
  - 1-选：在当前节点装摄像头
  - 2-不选：在父节点装摄像头
  - 3-不选：在左/右儿子装摄像头（至少一个儿子）
- 子树根节点
  - 情况1： `= min(左1，左2，左3)+min(右1，右2，右3)+1`
  - 情况2：`= min（左1，左3）+min（右1，右3）`
  - 情况3：`= min（左1+右3，左1+右1，左3+右1）`
- 最终答案
  - `min（根节点为1，根节点为3）`
    - 根节点没有父节点
- 边界
  - 空节点：1为无穷大（不能装监控），2、3为0（不需要被监控）

```python
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return inf, 0, 0
            l_ch, l_fa, l_son = dfs(node.left)
            r_ch, r_fa, r_son = dfs(node.right)
            return min(l_ch, l_fa, l_son) + min(r_ch, r_fa, r_son) + 1, \
            min(l_ch, l_son) + min(r_ch, r_son), \
            min(l_ch + r_son, l_ch + r_ch, l_son + r_ch)
        root_ch, _, root_son = dfs(root)
        return min(root_ch, root_son)
```

#### 变形1

在节点x安装监控需要花费cost[x]

```python
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node:
                return inf, 0, 0
            l_ch, l_fa, l_son = dfs(node.left)
            r_ch, r_fa, r_son = dfs(node.right)
            ### 1换成花费
            return min(l_ch, l_fa, l_son) + min(r_ch, r_fa, r_son) + cost[node], \
            min(l_ch, l_son) + min(r_ch, r_son), \
            min(l_ch + r_son, l_ch + r_ch, l_son + r_ch)
        root_ch, _, root_son = dfs(root)
        return min(root_ch, root_son)
```

#### 变形2

- 普通树（**情况3**需要枚举很多种情况）
  - 同样可以应用到二叉树的情况3的简化上
- 简化为：至少选一个情况1节点
  - `min(A1，A3) + min(B1，B3) + min(C1，C3) + ...`
  - 特殊情况：所有子节点的情况3**都小于**情况1
  - 补充一个偏置量：`+max(0, min(A1-A3, B1-B3, ...)`
    - 意义：后面的项大于0时，意味着满足特殊情况，所以需要选一个**情况1和情况3差值最小的**节点来**选择情况1**

# 70.爬楼梯

> 简单

# 118.杨辉三角

> 简单

i和j的定义要注意

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        ans = []

        for i in range(0, numRows):
            print(i)
            row = [0] * (i+1)
            row[0] = row[-1] = 1
            for j in range(1, i):
                row[j] = ans[i-1][j-1] + ans[i-1][j]
            print(row)
            ans.append(row)

        return ans
```

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        c = [[1] * (i + 1) for i in range(numRows)]
        for i in range(2, numRows):
            for j in range(1, i):
                # 左上方的数 + 正上方的数
                c[i][j] = c[i - 1][j - 1] + c[i - 1][j]
        return c
```

# 198.打家劫舍

> 中等

找时间再做一遍

# 279.完全平方数

> 中等

完全背包

### 递推

还是超时了

```python
class Solution:
    def numSquares(self, n: int) -> int:
        if n==1:
            return 1
        k = n // 2
        while k*k <= n:
            k += 1
        k -= 1
        
        dp = [[inf] * (n+1) for _ in range(k+1)]
        dp[0][0] = 0
		# 注意越界问题1（主要与k有关）
        for i in range(1, k+1):
            for c in range(n+1):
                x = i * i
                if c < x:
                    # 注意越界问题2
                    dp[i][c] = dp[i-1][c]
                else:
                    dp[i][c] = min(dp[i-1][c], dp[i][c - x] + 1)
        
        return dp[k][n]
```

### 双数组

也超时

### 单数组

AC

```python
class Solution:
    def numSquares(self, n: int) -> int:
        if n==1:
            return 1
        k = n // 2
        while k*k <= n:
            k += 1
        k -= 1
        
        dp = [inf] * (n+1)
        dp[0] = 0

        for i in range(1, k+1):
            x = i * i
            for c in range(x, n+1):
                dp[c] = min(dp[c], dp[c - x] + 1)
        
        return dp[n]
```



# 322.零钱兑换



找时间再做

# 139.单词拆分

> 中等

- 思路：选不选当前的字符
  - 可以选的前提条件：当前子串在词典中

### 记忆化搜索

我的代码

```python
### 顺序
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        
        @cache
        def dfs(i, j):
            if i >= n:
                return True
            elif j >= n:
                return False
            t = s[i:j+1]
            if t in wordDict:
                return dfs(j+1, j+1) or dfs(i, j+1)
            return dfs(i, j+1)
        
        return dfs(0, 0)
### 逆序
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)

        @cache
        def dfs(i, j):
            if j < 0:
                return True
            elif i < 0:
                return False
            t = s[i:j+1]
            if t in wordDict:
                return dfs(i-1, i-1) or dfs(i-1, j)
            return dfs(i-1, j)
        
        return dfs(n-1, n-1)
```



### 递推

> 逆序递归更容易翻译成顺序的递推

- 我的代码：开始以为边界条件没写对
  - 实际上：选择**子串时下标**有问题

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        
        dp = [[False] * (n+1) for _ in range(n+1)]
        dp[0][0] = True

        for i in range(1, n+1):
            for j in range(i, n+1):
                ### key
                t = s[i-1:j]
                if t in wordDict:
                    dp[i][j] = dp[i-1][i-1] or dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[n][n]
```

#### 一个数组

- 直接去掉：因为条件是`True or False`

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        
        dp = [False] * (n+1)
        dp[0] = True

        for i in range(1, n+1):
            for j in range(i, n+1):
                t = s[i-1:j]
                if t in wordDict:
                    dp[j] = dp[i-1] or dp[j]
                else:
                    dp[j] = dp[j]

        return dp[n]
```

- **可再优化：**
  - **利用哈希集合记录wordDict**
  - **记录最大的子串长度，可以提前跳出**

### 另一种写法

#### 记忆化搜索

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        max_len = max(map(len, wordDict))  # 用于限制下面 j 的循环次数
        words = set(wordDict)  # 便于快速判断 s[j:i] in words

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int) -> bool:
            if i == 0:  # 成功拆分！
                return True
            for j in range(i - 1, max(i - max_len - 1, -1), -1):
                if s[j:i] in words and dfs(j):
                    return True
            return False

        return dfs(len(s))
```

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        max_len = max(map(len, wordDict))  # 用于限制下面 j 的循环次数
        words = set(wordDict)  # 便于快速判断 s[j:i] in words

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int) -> bool:
            if i == 0:  # 成功拆分！
                return True
            return any(s[j:i] in words and dfs(j)
                       for j in range(i - 1, max(i - max_len - 1, -1), -1))

        return dfs(len(s))
```



#### 递推

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        max_len = max(map(len, wordDict))  # 用于限制下面 j 的循环次数
        words = set(wordDict)  # 便于快速判断 s[j:i] in words

        n = len(s)
        f = [True] + [False] * n
        for i in range(1, n + 1):
            for j in range(i - 1, max(i - max_len - 1, -1), -1):
                if f[j] and s[j:i] in words:
                    f[i] = True
                    break
        return f[n]
```

# 300. 最长递增子序列

> 中等

- 动态规划
- 贪心+二分查找

# 152. ==乘积==最大子数组（53的乘法版本）

> 中等

- 子数组（类比子串）

- 初始思路（想做记忆化搜索）
  - 对每个元素，算它和前面元素的最大乘积
    - 容易忽略的问题：
  - **超时**
    - 实际上是暴力解法
- 尝试1（与53类似的做法）：
  - 变成了贪心，无法得出正确的结果
  - 左边的乘积为负值时，有可能在接下来的过程中变为正值

- 主要难点
  - 如何在代码中实现**负负得正**对最大值的影响？
    - 遍历到**负值元素**时，当前元素与左边**子数组**的**==乘积最小值==**相乘才能得到**当前子数组最大值**
  - 因此解决思路为：分别记录子数组乘积的最大值和最小值
  - **错误的数组定义**：`dp_max = dp_min = [1] * n`
    - 指向同一个位置上的数组

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        dp_max = [1] * n
        dp_min = [1] * n
        dp_max[0] = dp_min[0] = nums[0]

        for i in range(1,n):
            max_mul = dp_max[i-1] * nums[i]
            min_mul = dp_min[i-1] * nums[i]
            dp_max[i] = max(max_mul, nums[i], min_mul)
            dp_min[i] = min(min_mul, nums[i], max_mul)
        
        return max(dp_max)
```

- 不加入数组的写法
  - 

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        ans = -inf  # 注意答案可能是负数
        f_max = f_min = 1
        for x in nums:
            f_max, f_min = max(f_max * x, f_min * x, x), \
                           min(f_max * x, f_min * x, x)
            ans = max(ans, f_max)
        return ans
```



## 53.最大子数组和（==前缀和/动态规划==）

> 中等

### 前缀和

- 思路
  - 把子数组的最大和转化为：
    - 两个前缀和的差的最大值

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = -inf
        pre_sum = min_pre_sum = 0
        for x in nums:
            pre_sum += x
            ans = max(ans, pre_sum - min_pre_sum)
            min_pre_sum = min(min_pre_sum, pre_sum)
        
        return ans
```

### 动态规划

- 思路
  - 当前元素是否和前面的子数组拼起来——前面的子数组和是负的，就不用要了
  - 代码关键点
    - dp数组记录的是单向的（如元素左边的子数组和情况），所以要取最大值
      - 想象一下元素右边子数组和为负值的情况
    - **不能直接拿最后一个**

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1], 0) + nums[i]
        ### 注意是返回最大的，而不是最后的
        return max(dp)
```

# 416. 分割等和子集

> 中等

- 初始思路
  - 数组**排序** + **前缀和**算法
    - 考虑不全面

- 解决思路：0-1背包

### 记忆化搜索

```python
from functools import cache

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        total /= 2
        n = len(nums)
        @cache
        def dfs(i, c):
            if c == 0:
                return True
            if i < 0:
                return False
            x = nums[i]
            
            return dfs(i-1, c-x) or dfs(i-1, c)
        
        return dfs(n-1, total)
```

### 递推

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        total //= 2
        n = len(nums)
        
        dp = [[False] * (total+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = True
        
        for i in range(1, n+1):
            x = nums[i-1]
            for c in range(total+1):
                if c < x:
                    dp[i][c] = dp[i-1][c]
                else:
                    dp[i][c] = dp[i-1][c] or dp[i-1][c - x]
        return dp[n][total]
```

- 一个数组
  - 倒序防覆盖

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        total //= 2
        n = len(nums)
        
        dp = [False] * (total+1)
        dp[0] = True
        
        for i in range(1, n+1):
            x = nums[i-1]
            for c in range(total, x-1, -1):
                dp[c] = dp[c] or dp[c - x]
        return dp[total]
```

# 32. 最长有效括号

> 困难

## 第一直觉：==索引栈==

- 初始直觉
  - 使用**索引栈**而不是**字符栈**：要求**连续**子串
    - 栈顶部存储的是**最近一次**匹配失败的最大索引值
    - 与当前索引值作差即可获得连续子串长度

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        ans = 0

        stack = []
        for i, c in enumerate(s):
            if stack and s[stack[-1]] == '(' and c == ')':
                stack.pop()
                if not stack:
                    ans = max(ans, i+1)
                else:
                    ans = max(ans, i - stack[-1])
            else:
                stack.append(i)
        
        return ans
```

## 动态规划

- 初始思路
  - 最后一对字符（两种情况）：
    - `s[i-1]=='('`，`s[i]==')'`
      - `dp[i]=dp[i-2] + 2`
    - `s[i-1]==')'`，`s[i]==')'`
      - ==`dp[i]=dp[i-1]+2+dp[i-dp[i-1]-2]`==
      - 当前的两个、前一个子串的结尾、前一个子串的再往前的结尾？？？
- 难点
  - dp数组如果直接设为背包
    - 这种做法对应的是有效括号的数量（不要求连续）
  - **dp[i]**：表示以i为索引**结尾**的括号，最长的有效括号。
  - 注意是结尾，因此如果是'(', 其索引在dp数组中是为0的。
    - 因此实际上递推关系中，dp数组的关系只与其前面的连续几个有关。
    - 故很自然的也需要一个 ans  用于更新最大长度。

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        ans = 0
        n = len(s)
        dp = [0] * n

        for i in range(1, n):
            if s[i] == ')':
                if s[i-1] == '(':
                    if i >= 2:
                        dp[i] = dp[i-2] + 2
                    else:
                        dp[i] = 2
                else:
                    if i - dp[i-1] > 0 and s[i - dp[i-1] - 1] == '(':
                        if i - dp[i-1] - 2 >= 0:
                            dp[i] = dp[i-1] + 2 + dp[i - dp[i-1] - 2]
                        else:
                            dp[i] = dp[i-1] + 2
                
                ans = max(ans, dp[i])
        return ans
```

