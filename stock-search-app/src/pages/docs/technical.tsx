import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Button,
  Link as MuiLink,
  Chip
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const TechnicalIndicatorOptimizationPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Technical Indicator Optimization for Indian Stocks | QuantPort India Docs</title>
        <meta name="description" content="Learn Technical Indicator Optimization for constructing optimal Indian stock portfolios using cross-sectional z-scores, linear programming, and NSE/BSE technical signals." />
        <meta property="og:title" content="Technical Indicator Optimization for Indian Stocks | QuantPort India Docs" />
        <meta property="og:description" content="Learn Technical Indicator Optimization for constructing optimal Indian stock portfolios using cross-sectional z-scores, linear programming, and NSE/BSE technical signals." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/technical" />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Docs
            </Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">
              ← Back to Portfolio Optimizer
            </Button>
          </Link>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Technical Indicator Optimization
            <Chip 
              label="NEW" 
              size="small" 
              sx={{ 
                backgroundColor: '#4caf50', 
                color: 'white', 
                fontWeight: 'bold',
                fontSize: '0.7rem',
                ml: 2
              }} 
            />
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Cross-sectional signal optimization using technical indicators
          </Typography>
        </Box>
        
        {/* Introduction */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Introduction
          </Typography>
          <Typography paragraph>
            Technical Indicator Optimization represents a paradigm shift from traditional return-based portfolio 
            optimization to signal-based optimization. Instead of using historical returns and covariance matrices, 
            this approach leverages <strong>cross-sectional z-scores</strong> of technical indicators to construct 
            portfolios that capitalize on relative strength and momentum patterns across assets.
          </Typography>
          <Typography paragraph>
            This methodology transforms the portfolio optimization problem from a quadratic programming problem 
            (as in Mean-Variance Optimization) into a <strong>Linear Programming (LP)</strong> problem, making 
            it computationally efficient and robust to estimation errors inherent in return forecasting.
          </Typography>
          <Typography paragraph>
            The approach is particularly well-suited for the Indian equity markets (NSE/BSE) where technical 
            analysis has historically shown strong predictive power, and where the cross-sectional dispersion 
            of technical signals can be effectively captured and monetized.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine you're a cricket scout looking for the best players for your team. Instead of just looking 
            at each player's individual statistics (like traditional portfolio optimization looks at individual 
            stock returns), you compare how each player performs <em>relative to their peers</em> across 
            multiple skills: batting average, bowling speed, fielding accuracy, etc.
          </Typography>
          <Typography paragraph>
            Technical Indicator Optimization works similarly. For each stock, we calculate multiple technical 
            indicators (RSI, Moving Averages, Williams %R, etc.), then determine how each stock ranks relative 
            to all other stocks for each indicator. A stock with consistently high relative rankings across 
            multiple indicators gets a higher allocation in the optimized portfolio.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> If RELIANCE has an RSI of 70 when the average RSI across all Nifty 
              stocks is 50 with a standard deviation of 15, then RELIANCE has a z-score of +1.33. If it also 
              has positive z-scores for momentum and moving average indicators, it receives a higher weight 
              in the optimized portfolio.
            </Typography>
          </Box>
        </Paper>
        
        {/* Technical Indicators */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Technical Indicators: Mathematical Foundations
          </Typography>
          <Typography paragraph>
            Our optimization framework incorporates 12 different technical indicators, each capturing different 
            aspects of price momentum, trend, and market psychology. Here are the mathematical formulations:
          </Typography>
          
          {/* Moving Averages */}
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            1. Simple Moving Average (SMA)
          </Typography>
          <Typography paragraph>
            The SMA smooths price data by calculating the arithmetic mean over a specified period:
          </Typography>
          <Equation math="SMA_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}" />
          <Typography paragraph>
            Where <InlineMath math="P_t" /> is the price at time <InlineMath math="t" /> and <InlineMath math="n" /> is the lookback period.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            2. Exponential Moving Average (EMA)
          </Typography>
          <Typography paragraph>
            EMA gives more weight to recent prices, making it more responsive to new information:
          </Typography>
          <Equation math="EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}" />
          <Typography paragraph>
            Where <InlineMath math="\alpha = \frac{2}{n+1}" /> is the smoothing factor.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            3. Weighted Moving Average (WMA)
          </Typography>
          <Typography paragraph>
            WMA assigns linearly decreasing weights to older prices:
          </Typography>
          <Equation math="WMA_n(t) = \frac{\sum_{i=0}^{n-1} (n-i) \cdot P_{t-i}}{\sum_{i=0}^{n-1} (n-i)}" />
          
          {/* Momentum Oscillators */}
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            4. Relative Strength Index (RSI)
          </Typography>
          <Typography paragraph>
            RSI measures the magnitude of price changes to evaluate overbought/oversold conditions:
          </Typography>
          <Equation math="RSI = 100 - \frac{100}{1 + RS}" />
          <Typography paragraph>
            Where <InlineMath math="RS = \frac{Average\ Gain}{Average\ Loss}" /> over the specified period.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            5. Williams %R
          </Typography>
          <Typography paragraph>
            Williams %R compares the current close to the high-low range over a lookback period:
          </Typography>
          <Equation math="\%R = \frac{Highest\ High - Close}{Highest\ High - Lowest\ Low} \times (-100)" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            6. Commodity Channel Index (CCI)
          </Typography>
          <Typography paragraph>
            CCI measures the variation of price from its statistical mean:
          </Typography>
          <Equation math="CCI = \frac{Typical\ Price - SMA(Typical\ Price)}{0.015 \times Mean\ Deviation}" />
          <Typography paragraph>
            Where <InlineMath math="Typical\ Price = \frac{High + Low + Close}{3}" />
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            7. Rate of Change (ROC)
          </Typography>
          <Typography paragraph>
            ROC measures the percentage change in price over a specified period:
          </Typography>
          <Equation math="ROC_n = \frac{P_t - P_{t-n}}{P_{t-n}} \times 100" />
          
          {/* Volatility Indicators */}
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            8. Average True Range (ATR)
          </Typography>
          <Typography paragraph>
            ATR measures market volatility by decomposing the entire range of an asset price:
          </Typography>
          <Equation math="TR = \max(High - Low, |High - Close_{prev}|, |Low - Close_{prev}|)" />
          <Equation math="ATR = SMA(TR, n)" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            9. SuperTrend
          </Typography>
          <Typography paragraph>
            SuperTrend combines ATR with price to create dynamic support/resistance levels:
          </Typography>
          <Equation math="Basic\ Upper\ Band = \frac{High + Low}{2} + (Multiplier \times ATR)" />
          <Equation math="Basic\ Lower\ Band = \frac{High + Low}{2} - (Multiplier \times ATR)" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            10. Bollinger Bands
          </Typography>
          <Typography paragraph>
            Bollinger Bands use standard deviation to create dynamic bands around a moving average:
          </Typography>
          <Equation math="Upper\ Band = SMA + (k \times \sigma)" />
          <Equation math="Lower\ Band = SMA - (k \times \sigma)" />
          <Typography paragraph>
            Where <InlineMath math="k" /> is typically 2 and <InlineMath math="\sigma" /> is the standard deviation.
          </Typography>
          
          {/* Volume Indicators */}
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            11. On-Balance Volume (OBV)
          </Typography>
          <Typography paragraph>
            OBV combines price and volume to show buying/selling pressure:
          </Typography>
          <Equation math="OBV_t = \begin{cases} 
            OBV_{t-1} + Volume_t & \text{if } Close_t > Close_{t-1} \\
            OBV_{t-1} - Volume_t & \text{if } Close_t < Close_{t-1} \\
            OBV_{t-1} & \text{if } Close_t = Close_{t-1}
          \end{cases}" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            12. Accumulation/Distribution (A/D)
          </Typography>
          <Typography paragraph>
            A/D line shows the relationship between price and volume flow:
          </Typography>
          <Equation math="Money\ Flow\ Multiplier = \frac{(Close - Low) - (High - Close)}{High - Low}" />
          <Equation math="A/D = Previous\ A/D + (Money\ Flow\ Multiplier \times Volume)" />
        </Paper>
        
        {/* Cross-Sectional Z-Score Methodology */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Cross-Sectional Z-Score Methodology
          </Typography>
          <Typography paragraph>
            The core innovation of Technical Indicator Optimization lies in transforming absolute indicator 
            values into relative rankings through cross-sectional standardization.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 1: Calculate Technical Indicators
          </Typography>
          <Typography paragraph>
            For each asset <InlineMath math="i" /> and each technical indicator <InlineMath math="j" />, 
            we calculate the indicator value <InlineMath math="I_{i,j}(t)" /> at time <InlineMath math="t" />.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 2: Cross-Sectional Standardization
          </Typography>
          <Typography paragraph>
            At each time period <InlineMath math="t" />, we standardize each indicator across all assets:
          </Typography>
          <Equation math="Z_{i,j}(t) = \frac{I_{i,j}(t) - \mu_j(t)}{\sigma_j(t)}" />
          <Typography paragraph>
            Where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="Z_{i,j}(t)" />: Z-score of asset <InlineMath math="i" /> for indicator <InlineMath math="j" /> at time <InlineMath math="t" />
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\mu_j(t) = \frac{1}{N} \sum_{i=1}^{N} I_{i,j}(t)" />: Cross-sectional mean of indicator <InlineMath math="j" />
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\sigma_j(t) = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (I_{i,j}(t) - \mu_j(t))^2}" />: Cross-sectional standard deviation
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 3: Signal Aggregation
          </Typography>
          <Typography paragraph>
            We combine multiple z-scores to create a composite signal for each asset:
          </Typography>
          <Equation math="S_i(t) = \sum_{j=1}^{M} w_j \cdot Z_{i,j}(t)" />
          <Typography paragraph>
            Where <InlineMath math="w_j" /> are the weights assigned to each indicator (equal weighting by default: <InlineMath math="w_j = \frac{1}{M}" />).
          </Typography>
        </Paper>
        
        {/* Linear Programming Formulation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Linear Programming Formulation
          </Typography>
          <Typography paragraph>
            The technical indicator optimization problem is formulated as a Linear Programming problem 
            that maximizes the expected portfolio signal while satisfying practical constraints.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Objective Function
          </Typography>
          <Typography paragraph>
            We maximize the portfolio's expected signal strength:
          </Typography>
          <Equation math="\max_{w} \sum_{i=1}^{N} w_i \cdot S_i" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Constraints
          </Typography>
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>1. Budget Constraint</strong>
          </Typography>
          <Typography paragraph>
            The portfolio weights must sum to 1 (fully invested):
          </Typography>
          <Equation math="\sum_{i=1}^{N} w_i = 1" />
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>2. Long-Only Constraint</strong>
          </Typography>
          <Typography paragraph>
            No short-selling is allowed:
          </Typography>
          <Equation math="w_i \geq 0 \quad \forall i" />
          
          <Typography variant="subtitle1" gutterBottom>
            <strong>3. Maximum Weight Constraint</strong>
          </Typography>
          <Typography paragraph>
            To ensure diversification, we limit individual asset weights:
          </Typography>
          <Equation math="w_i \leq w_{max} \quad \forall i" />
          <Typography paragraph>
            Where <InlineMath math="w_{max}" /> is typically set to 30-40% to prevent concentration risk.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Complete Mathematical Formulation
          </Typography>
          <Equation math="\begin{aligned} 
            \max_{w} \quad & \sum_{i=1}^{N} w_i \cdot S_i \\
            \text{subject to} \quad & \sum_{i=1}^{N} w_i = 1 \\
            & w_i \geq 0 \quad \forall i \\
            & w_i \leq w_{max} \quad \forall i
          \end{aligned}" />
          
          <Typography paragraph sx={{ mt: 2 }}>
            This LP formulation is solved using standard linear programming solvers, ensuring optimal 
            allocation based on the strength of technical signals across the investment universe.
          </Typography>
        </Paper>
        
        {/* Advantages and Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages and Implementation Considerations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Computational Efficiency:</strong> Linear programming is computationally 
                      faster and more stable than quadratic programming used in traditional MVO.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Robust to Estimation Errors:</strong> Avoids the need to estimate 
                      expected returns and covariance matrices, which are notoriously difficult to predict.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Cross-Sectional Focus:</strong> Captures relative strength patterns 
                      that are often more persistent than absolute return predictions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Multiple Signal Integration:</strong> Systematically combines multiple 
                      technical indicators to reduce noise and false signals.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Adaptability:</strong> Framework can easily incorporate new technical 
                      indicators or adjust indicator weights based on market conditions.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Implementation Considerations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Look-ahead Bias:</strong> Must ensure all technical indicators use only 
                      historical data available at the time of portfolio construction.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Transaction Costs:</strong> High signal turnover may lead to excessive 
                      trading costs; consider implementing turnover constraints.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Market Regime Changes:</strong> Technical signals may lose effectiveness 
                      during structural market changes or unusual market conditions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Overfitting Risk:</strong> Using too many indicators or complex combinations 
                      may lead to overfitting to historical data.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Beta Considerations:</strong> Unlike traditional optimization, this approach 
                      doesn't explicitly control for market beta or factor exposures.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Performance Characteristics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Performance Characteristics for Indian Markets
          </Typography>
          <Typography paragraph>
            Technical Indicator Optimization has shown particular effectiveness in Indian equity markets 
            due to several unique characteristics:
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Market Microstructure Benefits
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Momentum Persistence:</strong> Indian markets exhibit stronger momentum effects 
                compared to developed markets, making technical signals more predictive.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Cross-Sectional Dispersion:</strong> High dispersion in stock performance 
                within sectors creates opportunities for relative strength strategies.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Retail Participation:</strong> Significant retail investor participation leads 
                to behavioral patterns that technical indicators can effectively capture.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Risk Management Features
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Adaptive Allocation:</strong> Weights adjust automatically based on changing 
                signal strength, providing dynamic risk management.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Diversification:</strong> Maximum weight constraints ensure no single position 
                dominates the portfolio.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Control:</strong> Quick signal adaptation helps reduce portfolio 
                drawdowns during adverse market conditions.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Jegadeesh, N., & Titman, S. (1993)</strong>. "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." 
                <em>The Journal of Finance</em>, 48(1), 65-91.
                <MuiLink href="https://doi.org/10.1111/j.1540-6261.1993.tb04702.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Lo, A. W., Mamaysky, H., & Wang, J. (2000)</strong>. "Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation." 
                <em>The Journal of Finance</em>, 55(4), 1705-1765.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Park, C. H., & Irwin, S. H. (2007)</strong>. "What Do We Know About the Profitability of Technical Analysis?" 
                <em>Journal of Economic Surveys</em>, 21(4), 786-826.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)</strong>. "Time Series Momentum." 
                <em>Journal of Financial Economics</em>, 104(2), 228-250.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2013)</strong>. "Demystifying Managed Futures." 
                <em>Journal of Investment Management</em>, 11(3), 42-58.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Balvers, R., & Wu, Y. (2006)</strong>. "Momentum and Mean Reversion Across National Equity Markets." 
                <em>Journal of Empirical Finance</em>, 13(1), 24-48.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Wilder, J. W. (1978)</strong>. <em>New Concepts in Technical Trading Systems</em>. Trend Research.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Bollinger, J. (2001)</strong>. <em>Bollinger on Bollinger Bands</em>. McGraw-Hill Education.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Alvior, E. B. (2021)</strong>. "Moving Average Indicator and Trade Set-up as Correlates to Investment Trading in Stock Market." 
                <em>Journal of Business and Management Studies</em>, 3(2), 140-151.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Belantari, A. (2024)</strong>. "An Optimization Enhanced Technical Momentum Method: Part One-Equities Portfolio." 
                <em>Medium Technical Analysis Research</em>.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Mean-Variance Optimization
                </Typography>
                <Typography variant="body2" paragraph>
                  The traditional approach to portfolio optimization based on expected returns and covariance.
                </Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Minimum Volatility
                </Typography>
                <Typography variant="body2" paragraph>
                  Risk-focused optimization that minimizes portfolio volatility without signal considerations.
                </Typography>
                <Link href="/docs/min-vol" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Hierarchical Risk Parity
                </Typography>
                <Typography variant="body2" paragraph>
                  Machine learning approach using clustering for portfolio construction.
                </Typography>
                <Link href="/docs/hrp" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Add getStaticProps to generate this page at build time
export const getStaticProps = async () => {
  return {
    props: {},
  };
};

export default TechnicalIndicatorOptimizationPage; 