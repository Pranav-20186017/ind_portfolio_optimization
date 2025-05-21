import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ReferenceLine, ResponsiveContainer, 
  Label, Scatter
} from 'recharts';

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

// Define all chart data points
const chartData = [
  { beta: 0, expectedReturn: 3 },  // Risk-free rate (3%)
  { beta: 0.8, expectedReturn: 10, name: 'Asset A (Undervalued)' }, // Asset A
  { beta: 1.2, expectedReturn: 7, name: 'Asset B (Overvalued)' },  // Asset B
  { beta: 1.5, expectedReturn: 10.5, name: 'Asset C (Correctly Priced)' }, // Asset C
  { beta: 2, expectedReturn: 13 }  // Extrapolated market return
];

// Define SML data (just the line)
const smlData = [
  { beta: 0, expectedReturn: 3 },
  { beta: 2, expectedReturn: 13 }
];

// Sample assets colors
const assetColors = {
  'Asset A (Undervalued)': '#82ca9d',
  'Asset B (Overvalued)': '#ff8042',
  'Asset C (Correctly Priced)': '#8884d8'
};

const CapitalAssetPricingModelPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Capital Asset Pricing Model (CAPM) | Portfolio Optimization</title>
        <meta name="description" content="Learn about the Capital Asset Pricing Model (CAPM), the cornerstone of modern financial theory that helps investors understand the relationship between systematic risk and expected return." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Education
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
            Capital Asset Pricing Model (CAPM)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Understanding the relationship between risk and expected return
          </Typography>
        </Box>
        
        {/* Introduction Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Introduction
          </Typography>
          <Typography paragraph>
            The <strong>Capital Asset Pricing Model (CAPM)</strong> is a cornerstone of modern financial theory developed independently by William Sharpe, John Lintner, and Jan Mossin in the 1960s, building on Harry Markowitz's work on portfolio theory. For this contribution, Sharpe shared the 1990 Nobel Prize in Economics with Markowitz and Merton Miller.
          </Typography>
          <Typography paragraph>
            CAPM provides a framework to understand the relationship between systematic risk and expected return for assets, particularly stocks. It's widely used in finance for pricing risky securities, generating expected returns, and evaluating investment performance. The model introduces the concept that investors need to be compensated in two ways: the time value of money and risk.
          </Typography>
          <Typography paragraph>
            Despite its theoretical elegance and practical utility, CAPM has been challenged by empirical tests, leading to the development of more complex models. Nevertheless, it remains widely taught and used due to its simplicity and powerful insights into the nature of risk and return.
          </Typography>
        </Paper>
        
        {/* Big-Picture Intuition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Big-Picture Intuition
          </Typography>
          <Typography paragraph>
            CAPM is often described as the finance world's <strong>"one-factor model."</strong> At its core, it presents a revolutionary idea: investors are only compensated for taking <strong>systematic (market-wide) risk</strong>, not for specific risks that can be eliminated through diversification.
          </Typography>
          <Typography paragraph>
            In essence, CAPM distinguishes between two types of risk:
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Systematic Risk (Market Risk)
                  </Typography>
                                    <Box component="div" sx={{ ml: 2 }}>                    <Typography variant="body2" paragraph>• Cannot be eliminated through diversification</Typography>                    <Typography variant="body2" paragraph>• Affects the entire market (e.g., economic cycles, interest rates)</Typography>                    <Typography variant="body2" paragraph>• Investors are rewarded with a risk premium for bearing this risk</Typography>                    <Typography variant="body2" paragraph>• Measured by beta (β)</Typography>                  </Box>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Unsystematic Risk (Idiosyncratic Risk)
                  </Typography>
                                    <Box component="div" sx={{ ml: 2 }}>                    <Typography variant="body2" paragraph>• Can be eliminated through diversification</Typography>                    <Typography variant="body2" paragraph>• Specific to individual stocks or sectors</Typography>                    <Typography variant="body2" paragraph>• No expected premium for bearing this risk</Typography>                    <Typography variant="body2" paragraph>• Examples: company scandals, management changes, product failures</Typography>                  </Box>
                </Box>
              </Grid>
            </Grid>
          </Box>
          <Typography paragraph>
            This distinction leads to the powerful conclusion that diversification is the "free lunch" of investing—it eliminates specific risks without reducing expected returns. What remains is systematic risk, which becomes the only relevant risk factor in pricing assets.
          </Typography>
        </Paper>
        
        {/* The Core Equation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            The Core Equation
          </Typography>
          <Typography paragraph>
            The CAPM equation elegantly expresses the expected return of an asset as a function of the risk-free rate plus a risk premium:
          </Typography>
          <Equation math="\boxed{\;\mathbb{E}[R_i] \;=\; R_f \;+\;\beta_i \,\bigl(\mathbb{E}[R_m] - R_f\bigr)\;}" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Components of the CAPM Formula
          </Typography>
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Term</strong></TableCell>
                  <TableCell><strong>Meaning</strong></TableCell>
                  <TableCell><strong>Practical Interpretation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\mathbb{E}[R_i]" />
                  </TableCell>
                  <TableCell>Expected return of asset/portfolio <em>i</em></TableCell>
                  <TableCell>The return investors should require for investing in the asset</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="R_f" />
                  </TableCell>
                  <TableCell>Risk-free rate</TableCell>
                  <TableCell>Typically yields from T-bills, government securities, or overnight repo rates</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\mathbb{E}[R_m]" />
                  </TableCell>
                  <TableCell>Expected return of the market portfolio</TableCell>
                  <TableCell>Return of a broad market index like NIFTY 50 or S&P 500</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\beta_i" />
                  </TableCell>
                  <TableCell>Asset's market beta (sensitivity to market movements)</TableCell>
                  <TableCell>Measure of systematic risk; how much the asset's returns move with the market</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\mathbb{E}[R_m] - R_f" />
                  </TableCell>
                  <TableCell>Market risk premium</TableCell>
                  <TableCell>Extra return investors expect for taking on market risk rather than investing in risk-free assets</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            This straight-line relationship between beta and expected return is known as the <strong>Security Market Line (SML)</strong>. It represents the central insight of CAPM: higher beta (more systematic risk) should lead to higher expected returns.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Beta Definition
          </Typography>
          <Typography paragraph>
            Beta is calculated as:
          </Typography>
          <Equation math="\beta_i=\frac{\operatorname{Cov}(R_i,R_m)}{\operatorname{Var}(R_m)}" />
          <Typography paragraph>
            Or equivalently:
          </Typography>
          <Equation math="\beta_i = \rho_{i,m} \frac{\sigma_i}{\sigma_m}" />
          
                    <Typography paragraph>            Where:          </Typography>          <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1, ml: 2 }}>            <Typography component="div">              • <InlineMath math="\operatorname{Cov}(R_i,R_m)" /> is the covariance between the asset's returns and market returns            </Typography>            <Typography component="div">              • <InlineMath math="\operatorname{Var}(R_m)" /> is the variance of market returns            </Typography>            <Typography component="div">              • <InlineMath math="\rho_{i,m}" /> is the correlation coefficient between the asset and market            </Typography>            <Typography component="div">              • <InlineMath math="\sigma_i" /> is the standard deviation of the asset's returns            </Typography>            <Typography component="div">              • <InlineMath math="\sigma_m" /> is the standard deviation of market returns            </Typography>          </Box>
        </Paper>
        
        {/* Assumptions */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Assumptions
          </Typography>
          <Typography paragraph>
            The classical CAPM rests on several key assumptions that simplify the complexities of real markets:
          </Typography>
          
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Classical CAPM Assumptions
            </Typography>
            <Grid container spacing={2}>
                            <Grid item xs={12} md={6}>                <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>                  <Box component="div" sx={{ display: 'flex', gap: 1 }}>                    <Typography component="span">1.</Typography>                    <Typography component="div">                      <strong>Mean-variance investors</strong> — Investors care only about expected return and variance of their portfolios.                    </Typography>                  </Box>                  <Box component="div" sx={{ display: 'flex', gap: 1 }}>                    <Typography component="span">2.</Typography>                    <Typography component="div">                      <strong>Single-period horizon</strong> — All investors have the same evaluation period.                    </Typography>                  </Box>                  <Box component="div" sx={{ display: 'flex', gap: 1 }}>                    <Typography component="span">3.</Typography>                    <Typography component="div">                      <strong>Homogeneous expectations</strong> — All investors foresee the same expected returns and covariance matrix.                    </Typography>                  </Box>                </Box>              </Grid>              <Grid item xs={12} md={6}>                <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>                  <Box component="div" sx={{ display: 'flex', gap: 1 }}>                    <Typography component="span">4.</Typography>                    <Typography component="div">                      <strong>Perfect markets</strong> — No taxes, trading frictions, or short-sale constraints.                    </Typography>                  </Box>                  <Box component="div" sx={{ display: 'flex', gap: 1 }}>                    <Typography component="span">5.</Typography>                    <Typography component="div">                      <strong>Unlimited lending/borrowing at <InlineMath math="R_f" /></strong> — Everyone can leverage or de-leverage at the same risk-free rate.                    </Typography>                  </Box>                </Box>
              </Grid>
            </Grid>
          </Box>
          
          <Typography paragraph>
            Real financial markets violate these assumptions to varying degrees, which has led to the development of extensions like the Black CAPM (zero-beta CAPM), multi-factor models (Fama-French), and models incorporating liquidity premiums.
          </Typography>
        </Paper>
        
        {/* Derivation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Deriving the Formula
          </Typography>
          <Typography paragraph>
            The derivation of CAPM follows from Markowitz's Modern Portfolio Theory and reveals why only systematic risk matters:
          </Typography>
          
                    <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2 }}>            <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">•</Typography>                <Typography component="div">                  <strong>Start with the Efficient Frontier</strong> under Markowitz optimization.                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">•</Typography>                <Typography component="div">                  <strong>Add a risk-free asset</strong> to obtain the <strong>Capital Market Line (CML)</strong>, which represents all efficient portfolios combining the risk-free asset and a risky portfolio.                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">•</Typography>                <Typography component="div">                  <strong>The market portfolio</strong> sits at the tangency point of the CML with the efficient frontier. This represents all investable assets, weighted by their market value.                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">•</Typography>                <Typography component="div">                  <strong>Any efficient portfolio is a mix</strong> of the risk-free asset and the market portfolio.                </Typography>              </Box>            </Box>          </Box>
          
          <Typography paragraph>
            From these insights, we can derive the CAPM equation for any asset <InlineMath math="i" />:
          </Typography>
          <Equation math="\mathbb{E}[R_i] - R_f \;=\; \beta_i \bigl(\mathbb{E}[R_m] - R_f\bigr)" />
          
          <Typography paragraph>
            This shows that expected excess return is <strong>proportional to covariance with the market</strong>, not to the asset's total variance. This is the fundamental insight of CAPM—only systematic risk is priced in equilibrium.
          </Typography>
        </Paper>
        
        {/* Estimating CAPM in Practice */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Estimating CAPM in Practice
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            OLS Regression Approach
          </Typography>
          <Typography paragraph>
            In practice, CAPM parameters are typically estimated using Ordinary Least Squares (OLS) regression:
          </Typography>
          <Equation math="R_{i,t}-R_{f,t} \;=\; \alpha_i \;+\; \beta_i(R_{m,t}-R_{f,t}) \;+\; \varepsilon_t" />
          
                    <Typography paragraph>            Where:          </Typography>          <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1, ml: 2 }}>            <Typography component="div">              • <InlineMath math="R_{i,t}-R_{f,t}" /> is the excess return of asset <InlineMath math="i" /> at time <InlineMath math="t" />            </Typography>            <Typography component="div">              • <InlineMath math="R_{m,t}-R_{f,t}" /> is the excess return of the market at time <InlineMath math="t" />            </Typography>            <Typography component="div">              • <InlineMath math="\alpha_i" /> is the intercept, representing Jensen's Alpha (abnormal return)            </Typography>            <Typography component="div">              • <InlineMath math="\beta_i" /> is the slope coefficient, representing the asset's beta            </Typography>            <Typography component="div">              • <InlineMath math="\varepsilon_t" /> is the error term            </Typography>          </Box>
          
                    <Typography paragraph>            In matrix form, the OLS estimator is <InlineMath math="\hat{\theta}=(X^{\!\top}X)^{-1}X^{\!\top}y" />, which produces:          </Typography>          <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1, ml: 2 }}>            <Typography component="div">              • <InlineMath math="\hat{\beta}_i" /> — market risk estimate            </Typography>            <Typography component="div">              • <InlineMath math="\hat{\alpha}_i" /> — abnormal return estimate (Jensen's Alpha)            </Typography>            <Typography component="div">              • <InlineMath math="R^2" /> — fraction of variance explained by the market            </Typography>          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Required Inputs
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Component</strong></TableCell>
                  <TableCell><strong>Typical Data Source</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Risk-free rate (<InlineMath math="R_f" />)</TableCell>
                  <TableCell>3-month T-bill or 10-year G-Sec yield (annualized)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Market return (<InlineMath math="R_m" />)</TableCell>
                  <TableCell>Broad index (NIFTY 50, S&P 500, MSCI World, etc.)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Return frequency</TableCell>
                  <TableCell>Daily or monthly; market and asset returns should use the same frequency</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ p: 2, mt: 3, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Note: Implementation in Our Portfolio Optimizer</strong>
            </Typography>
                                    <Typography variant="body2">              Our portfolio optimization backend automatically:            </Typography>            <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1, ml: 2, mt: 1 }}>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">1.</Typography>                <Typography component="div">                  Aligns dates between the portfolio and benchmark returns                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">2.</Typography>                <Typography component="div">                  Builds daily excess return series                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">3.</Typography>                <Typography component="div">                  Runs OLS regression using statsmodels                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">4.</Typography>                <Typography component="div">                  Produces rolling year-by-year betas to analyze beta stability                </Typography>              </Box>              <Box component="div" sx={{ display: 'flex', gap: 1 }}>                <Typography component="span">5.</Typography>                <Typography component="div">                  Calculates confidence intervals and p-values for statistical significance                </Typography>              </Box>            </Box>
          </Box>
        </Paper>
        
        {/* Reading the Security Market Line */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            The Security Market Line
          </Typography>
          <Typography paragraph>
            The Security Market Line (SML) is a graphical representation of the CAPM relationship. It plots expected excess returns against beta:
          </Typography>
          
          <Box sx={{ p: 3, bgcolor: '#f8f9fa', border: '1px dashed #ccc', borderRadius: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Security Market Line
            </Typography>
            
            <Box sx={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <LineChart
                  data={chartData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="beta" 
                    domain={[0, 2.2]} 
                    tickCount={6}
                    type="number"
                  >
                    <Label value="Beta (β)" offset={-5} position="insideBottom" />
                  </XAxis>
                  
                  <YAxis 
                    domain={[0, 15]} 
                    tickCount={6}
                    tickFormatter={(value) => `${value}%`}
                  >
                    <Label value="Expected Return" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
                  </YAxis>
                  
                  <Tooltip 
                    formatter={(value, name) => {
                      if (name === 'expectedReturn') {
                        return [`${value}%`, 'Expected Return'];
                      }
                      return [value, name];
                    }}
                    labelFormatter={(value) => `Beta: ${value}`}
                  />
                  
                  <Legend />
                  
                  {/* The SML line */}
                  <Line 
                    data={smlData}
                    type="monotone" 
                    dataKey="expectedReturn" 
                    stroke="#8884d8" 
                    name="Security Market Line"
                    strokeWidth={2}
                    dot={false}
                  />
                  
                  {/* Reference points */}
                  <ReferenceLine x={1} stroke="#666" strokeDasharray="3 3" label={{ value: 'Market', position: 'top' }} />
                  
                  {/* Asset A point */}
                  <Line
                    data={[chartData[1]]}
                    type="monotone"
                    dataKey="expectedReturn"
                    stroke="none"
                    name="Asset A: β=0.8, E(R)=10% (Undervalued)"
                    dot={{
                      r: 8,
                      fill: assetColors['Asset A (Undervalued)'],
                      strokeWidth: 1
                    }}
                  />
                  
                  {/* Asset B point */}
                  <Line
                    data={[chartData[2]]}
                    type="monotone"
                    dataKey="expectedReturn"
                    stroke="none"
                    name="Asset B: β=1.2, E(R)=7% (Overvalued)"
                    dot={{
                      r: 8,
                      fill: assetColors['Asset B (Overvalued)'],
                      strokeWidth: 1
                    }}
                  />
                  
                  {/* Asset C point */}
                  <Line
                    data={[chartData[3]]}
                    type="monotone"
                    dataKey="expectedReturn"
                    stroke="none"
                    name="Asset C: β=1.5, E(R)=10.5% (Correctly Priced)"
                    dot={{
                      r: 8,
                      fill: assetColors['Asset C (Correctly Priced)'],
                      strokeWidth: 1
                    }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
            
            {/* Example points explanation */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', mt: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mx: 2, my: 1 }}>
                <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: assetColors['Asset A (Undervalued)'], mr: 1 }} />
                <Typography variant="body2">Asset A: β=0.8, E(R)=10% (Undervalued)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mx: 2, my: 1 }}>
                <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: assetColors['Asset B (Overvalued)'], mr: 1 }} />
                <Typography variant="body2">Asset B: β=1.2, E(R)=7% (Overvalued)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mx: 2, my: 1 }}>
                <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: assetColors['Asset C (Correctly Priced)'], mr: 1 }} />
                <Typography variant="body2">Asset C: β=1.5, E(R)=10.5% (Correctly Priced)</Typography>
              </Box>
            </Box>
          </Box>
          
          <Typography paragraph>
            The SML has several key interpretations:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#e8f5e9', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Assets On the Line
                </Typography>
                <Typography variant="body2">
                  Correctly priced assets should fall exactly on the SML, indicating they offer returns commensurate with their systematic risk.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#e3f2fd', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Assets Above the Line
                </Typography>
                <Typography variant="body2">
                  Assets plotting above the SML have positive alpha (α), suggesting they're undervalued and offer excess returns beyond what their systematic risk would predict.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#ffebee', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Assets Below the Line
                </Typography>
                <Typography variant="body2">
                  Assets plotting below the SML have negative alpha, suggesting they're overpriced and don't adequately compensate investors for the systematic risk taken.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Common Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Common Applications
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Use Case</strong></TableCell>
                  <TableCell><strong>How CAPM Helps</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Cost of equity (DCF, WACC)</strong></TableCell>
                  <TableCell>
                    <Typography component="div">
                      <InlineMath math="k_e = R_f + \beta (MRP)" />, where MRP is the Market Risk Premium. This estimates the required return for equity investors.
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Performance attribution</strong></TableCell>
                  <TableCell>
                    Decompose portfolio return into alpha (skill) + beta × market (exposure to market movements).
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Risk budgeting / hedging</strong></TableCell>
                  <TableCell>
                    Size positions in a portfolio to achieve a desired target beta, managing overall market exposure.
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Factor models baseline</strong></TableCell>
                  <TableCell>
                    CAPM serves as the "1-factor" benchmark before adding size, value, momentum, and other factors.
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Asset allocation</strong></TableCell>
                  <TableCell>
                    Helps determine expected returns for different asset classes based on their systematic risk.
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Limitations & Extensions */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Limitations & Extensions
          </Typography>
          
          <Typography paragraph>
            While elegant in theory, CAPM has several empirical challenges. Various extensions have been developed to address these limitations:
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Limitation</strong></TableCell>
                  <TableCell><strong>Extension/Solution</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Beta instability</strong> — Betas tend to vary over time</TableCell>
                  <TableCell>
                    Rolling betas, regime-switching models, Kalman filter estimation
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Non-normal returns</strong> — Asset returns often exhibit fat tails and skewness</TableCell>
                  <TableCell>
                    Downside beta, conditional value-at-risk (CVaR) models, quantile regressions
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Multiple priced risks</strong> — Market beta alone doesn't explain all variation in returns</TableCell>
                  <TableCell>
                    Fama-French 3/5-factor models, Carhart 4-factor model, APT (Arbitrage Pricing Theory)
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>No risk-free borrowing</strong> — Not all investors can borrow at the risk-free rate</TableCell>
                  <TableCell>
                    Black CAPM (Zero-beta CAPM) replaces the risk-free rate with a zero-beta portfolio
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Liquidity concerns</strong> — CAPM doesn't account for transaction costs or liquidity differences</TableCell>
                  <TableCell>
                    Liquidity-adjusted CAPM, Acharya-Pedersen model
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph sx={{ mt: 2 }}>
            Despite these limitations, CAPM remains a fundamental model in finance due to its simplicity, intuitive appeal, and practical insights into the relationship between risk and return.
          </Typography>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Key References
          </Typography>
          <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box component="div">
              <Typography component="div">
                <strong>Sharpe, W. F. (1964)</strong>. "Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk." <em>Journal of Finance</em>, 19(3), 425-442.
                <Button 
                  variant="text"
                  component="a"
                  href="https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1964.tb02865.x"
                  target="_blank"
                  sx={{ ml: 1, p: 0, minWidth: 'auto', textTransform: 'none' }}
                >
                  Access the paper
                </Button>
              </Typography>
            </Box>
            <Box component="div">
              <Typography component="div">
                <strong>Lintner, J. (1965)</strong>. "The Valuation of Risk Assets and the Selection of Risky Investments in Stock Portfolios and Capital Budgets." <em>Review of Economics & Statistics</em>, 47(1), 13-37.
              </Typography>
            </Box>
            <Box component="div">
              <Typography component="div">
                <strong>Mossin, J. (1966)</strong>. "Equilibrium in a Capital Asset Market." <em>Econometrica</em>, 34(4), 768-783.
              </Typography>
            </Box>
            <Box component="div">
              <Typography component="div">
                <strong>Black, F. (1972)</strong>. "Capital Market Equilibrium with Restricted Borrowing." <em>Journal of Business</em>, 45(3), 444-455.
              </Typography>
            </Box>
            <Box component="div">
              <Typography component="div">
                <strong>Bodie, Kane & Marcus</strong>. <em>Investments</em> (12 ed.), McGraw-Hill, 2021 – Ch. 9-10.
              </Typography>
            </Box>
            <Box component="div">
              <Typography component="div">
                <strong>Fama, E. F., & French, K. R. (2004)</strong>. "The Capital Asset Pricing Model: Theory and Evidence." <em>Journal of Economic Perspectives</em>, 18(3), 25-46.
                <Button 
                  variant="text"
                  component="a"
                  href="https://www.aeaweb.org/articles?id=10.1257/0895330042162430"
                  target="_blank"
                  sx={{ ml: 1, p: 0, minWidth: 'auto', textTransform: 'none' }}
                >
                  Access the paper
                </Button>
              </Typography>
            </Box>
          </Box>
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
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of risk-adjusted return that helps investors understand the return of an investment compared to its risk.
                </Typography>
                <Link href="/docs/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sortino Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A variation of the Sharpe Ratio that differentiates harmful volatility from total volatility by using downside deviation.
                </Typography>
                <Link href="/docs/sortino-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Modern Portfolio Theory
                </Typography>
                <Typography variant="body2" paragraph>
                  The theoretical framework that underlies CAPM, focusing on how risk-averse investors can construct portfolios to maximize returns.
                </Typography>
                <Link href="/docs/modern-portfolio-theory" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
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

export default CapitalAssetPricingModelPage; 