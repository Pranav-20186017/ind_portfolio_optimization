import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Divider
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

const HierarchicalRiskParityPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Hierarchical Risk Parity (HRP) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Hierarchical Risk Parity (HRP), a modern portfolio optimization method that uses clustering and machine learning techniques to build diversified portfolios." />
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
            Hierarchical Risk Parity (HRP)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A modern machine learning approach to portfolio optimization
          </Typography>
        </Box>
        
        {/* Overview Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            Hierarchical Risk Parity (HRP), introduced by Marcos Lopez de Prado in 2016, is a revolutionary portfolio optimization 
            algorithm designed to address key issues associated with classical methods such as Mean-Variance Optimization (MVO). 
            Unlike traditional methods that rely on covariance matrix inversion (often unstable in practice), HRP uses clustering 
            and hierarchical methods to produce stable, diversified portfolios.
          </Typography>
          <Typography paragraph>
            HRP combines insights from graph theory, machine learning, and quantitative finance to efficiently manage risk without 
            explicit covariance matrix inversion, making it robust against estimation errors and particularly valuable when dealing 
            with large asset universes.
          </Typography>
        </Paper>
        
        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Intuitive Explanation
          </Typography>
          <Typography paragraph>
            Imagine organizing a large family reunion. Instead of trying to plan individually for every family member (asset), 
            you group them into smaller, logical family groups (clusters). Each subgroup is managed separately, ensuring the 
            planning process remains straightforward and stable.
          </Typography>
          <Typography paragraph>
            In HRP:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                Assets are first grouped into similar clusters based on their returns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Within each cluster, allocations are made to balance risk effectively.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Clusters are then combined hierarchically, ensuring overall portfolio risk remains well-managed.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            This structured, step-by-step approach naturally leads to more robust, intuitive, and stable portfolio construction, 
            especially useful when dealing with large numbers of assets.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Example:</strong> Consider a portfolio containing tech stocks, utility companies, banks, and consumer goods. 
              Traditional methods might struggle with estimation errors when calculating precise correlations between all pairs. 
              HRP would first identify that tech stocks tend to move together, as do banks, and so on. It would then allocate within 
              each sector and finally across sectors, respecting the natural structure of the market.
            </Typography>
          </Box>
        </Paper>
        
        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Detailed Mathematical & Algorithmic Explanation
          </Typography>
          <Typography paragraph>
            HRP follows a systematic, three-step approach:
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Step 1: Hierarchical Clustering
          </Typography>
          <Typography paragraph>
            <strong>Objective:</strong> Group assets based on similarity, typically using a correlation-based distance measure:
          </Typography>
          <Equation math="d_{i,j} = \sqrt{ \frac{1 - \rho_{i,j}}{2} }" />
          
          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="d_{i,j}" /> is the distance between assets <InlineMath math="i" /> and <InlineMath math="j" />.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\rho_{i,j}" /> is the correlation between asset returns.
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            This step creates a hierarchical structure (dendrogram), identifying natural clusters of assets. The distance metric 
            transforms correlations into distances, where assets with higher correlation have smaller distances between them.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 2: Quasi-Diagonalization
          </Typography>
          <Typography paragraph>
            Assets are reordered based on the dendrogram to place similar assets close together, resulting in a covariance matrix 
            that exhibits a <strong>block-diagonal structure</strong>:
          </Typography>
          <Equation math="\Sigma' = \begin{bmatrix} \Sigma_{A} & 0 & \dots & 0 \\ 0 & \Sigma_{B} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \Sigma_{Z} \end{bmatrix}" />
          
          <Typography paragraph>
            where each <InlineMath math="\Sigma_{X}" /> corresponds to a distinct asset cluster.
          </Typography>
          
          <Typography paragraph>
            The quasi-diagonalization process rearranges the covariance matrix so that highly correlated assets are adjacent. 
            This reorganization of the covariance matrix is crucial for the final step.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Step 3: Recursive Bisection (Risk Allocation)
          </Typography>
          <Typography paragraph>
            The portfolio weights are allocated through a recursive procedure, splitting clusters and assigning weights inversely 
            proportional to cluster variance:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                At each split, allocate capital based on the inverse variance of the clusters:
              </Typography>
            </li>
          </ul>
          <Equation math="w_{A} = \frac{1/\sigma_{A}}{1/\sigma_{A} + 1/\sigma_{B}}, \quad w_{B} = 1 - w_{A}" />
          
          <Typography paragraph>
            where <InlineMath math="\sigma_{A}" /> and <InlineMath math="\sigma_{B}" /> are the variances of clusters A and B.
          </Typography>
          
          <Typography paragraph>
            Continue recursively within each cluster until individual assets are allocated. This recursive risk budgeting ensures 
            diversification without reliance on unstable matrix inversion methods.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Linkage Methods
          </Typography>
          <Typography paragraph>
            Different linkage methods can be used in the hierarchical clustering step, each with distinct properties:
          </Typography>
          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Single Linkage:</strong> Measures the distance between the closest members of clusters. Tends to form elongated clusters.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Complete Linkage:</strong> Measures the distance between the furthest members of clusters. Forms more compact, spherical clusters.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Average Linkage:</strong> Measures the average distance between all pairs of observations. Often provides a balance between the other methods.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Ward's Method:</strong> Minimizes variance within clusters. Often produces more even-sized clusters.
                </Typography>
              </li>
            </ul>
          </Box>
          
          <Typography paragraph>
            For portfolio optimization, single linkage is often an efficient default choice, though the optimal method may depend on the specific asset characteristics and market conditions.
          </Typography>
        </Paper>
        
        {/* Visual Representation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Visual Representation
          </Typography>
          <Typography paragraph>
            HRP's approach can be visualized through several key representations:
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Dendrogram (Clustering Tree)
          </Typography>
          <Typography paragraph>
            A dendrogram visually represents how assets are hierarchically clustered:
          </Typography>
          
          <Box sx={{ textAlign: 'center', my: 3 }}>
            {/* Placeholder for an image - in production, replace with actual image */}
            <Paper 
              elevation={0}
              sx={{ 
                width: '100%', 
                height: 300, 
                bgcolor: '#f0f0f0', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}
            >
              <Typography variant="body2" color="text.secondary">
                [Dendrogram Visualization Placeholder]
              </Typography>
            </Paper>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Dendrogram showing hierarchical clustering of assets based on their correlation distances
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Correlation Matrix Heatmap
          </Typography>
          <Typography paragraph>
            Visualization of the correlation matrix before and after quasi-diagonalization:
          </Typography>
          
          <Box sx={{ textAlign: 'center', my: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Paper 
                  elevation={0}
                  sx={{ 
                    width: '100%', 
                    height: 250, 
                    bgcolor: '#f0f0f0', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center' 
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    [Original Correlation Matrix Heatmap]
                  </Typography>
                </Paper>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Original correlation matrix
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper 
                  elevation={0}
                  sx={{ 
                    width: '100%', 
                    height: 250, 
                    bgcolor: '#f0f0f0', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center' 
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    [Quasi-Diagonalized Correlation Matrix Heatmap]
                  </Typography>
                </Paper>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Quasi-diagonalized correlation matrix showing block structure
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </Paper>
        
        {/* Comparison with Other Methods */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Comparison with Other Optimization Methods
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ bgcolor: '#f8f9fa', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Mean-Variance Optimization
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Relies on inverse covariance matrix</Typography></li>
                    <li><Typography paragraph>Highly sensitive to estimation errors</Typography></li>
                    <li><Typography paragraph>Tends to produce concentrated portfolios</Typography></li>
                    <li><Typography paragraph>Requires expected returns estimates</Typography></li>
                    <li><Typography paragraph>Computationally simpler but less robust</Typography></li>
                  </ul>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ bgcolor: '#f1f8e9', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Hierarchical Risk Parity
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Uses hierarchical clustering approach</Typography></li>
                    <li><Typography paragraph>Robust to estimation errors</Typography></li>
                    <li><Typography paragraph>Produces well-diversified portfolios</Typography></li>
                    <li><Typography paragraph>Works without expected returns</Typography></li>
                    <li><Typography paragraph>Computationally efficient for large portfolios</Typography></li>
                  </ul>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ bgcolor: '#f3e5f5', p: 2, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Minimum Volatility
                  </Typography>
                  <ul>
                    <li><Typography paragraph>Focuses solely on minimizing risk</Typography></li>
                    <li><Typography paragraph>Still uses covariance matrix inversion</Typography></li>
                    <li><Typography paragraph>Better than MVO but less robust than HRP</Typography></li>
                    <li><Typography paragraph>No need for expected returns</Typography></li>
                    <li><Typography paragraph>May still suffer from concentration</Typography></li>
                  </ul>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        </Paper>
        
        {/* Advantages Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages of Hierarchical Risk Parity
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
                      <strong>Robustness:</strong> Avoids instability caused by covariance matrix inversion.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Diversification:</strong> Automatically achieves meaningful diversification by recursive risk budgeting.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Scalability:</strong> Ideal for large portfolios, capable of handling numerous assets efficiently.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Stability:</strong> Minimizes sensitivity to errors in covariance estimates, a common issue in traditional methods.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>No Expected Returns:</strong> Does not require estimates of expected returns, removing a major source of error.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Out-of-Sample Performance:</strong> Often outperforms traditional methods when applied to data outside the estimation period.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Limitations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Lack of Return Consideration:</strong> Does not directly incorporate expected returns in the optimization process.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Linkage Method Dependence:</strong> Results can vary based on the chosen hierarchical clustering method.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Interpretability:</strong> The hierarchical structure, while logical, may be less transparent to some investors than direct risk-return trade-offs.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Newer Methodology:</strong> Lacks the decades of empirical research behind traditional methods like MVO.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Suboptimality:</strong> May not achieve the theoretically optimal portfolio when perfect estimates are available.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Use Cases */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Practical Applications and Use Cases
          </Typography>
          
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Large-Scale Asset Allocation
                </Typography>
                <Typography variant="body2">
                  HRP excels when dealing with large universes of assets, where traditional methods often break down due to 
                  estimation errors and computational constraints. Institutional investors managing hundreds or thousands of 
                  securities benefit particularly from HRP's scalability.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  High-Frequency Portfolio Rebalancing
                </Typography>
                <Typography variant="body2">
                  For strategies requiring frequent rebalancing, HRP's computational efficiency and robustness to noise make it 
                  well-suited for adapting to rapidly changing market conditions without producing drastically different allocations 
                  from small changes in input data.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Risk-Focused Passive Investing
                </Typography>
                <Typography variant="body2">
                  When the primary goal is risk management rather than return maximization, HRP provides a sophisticated alternative to 
                  traditional methods. It's particularly valuable for creating diversified funds that aim to capture market returns while 
                  minimizing unnecessary risk.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Lopez de Prado, Marcos (2016)</strong>. "Building Diversified Portfolios that Outperform Out-of-Sample." <em>Journal of Portfolio Management</em>, 42(4), 59-69.
                <MuiLink href="https://doi.org/10.3905/jpm.2016.42.4.059" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Lopez de Prado, Marcos (2018)</strong>. "Advances in Financial Machine Learning." <em>John Wiley & Sons</em>. ISBN: 978-1119482086.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Raffinot, Thomas (2017)</strong>. "Hierarchical Clustering-Based Asset Allocation." <em>The Journal of Portfolio Management</em>, 44(2), 89-99.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Bailey, David H. & Lopez de Prado, Marcos (2013)</strong>. "An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization." <em>Algorithms</em>, 6(1), 169-196.
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
                  The cornerstone of Modern Portfolio Theory that balances return and risk.
                </Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Minimum Volatility
                </Typography>
                <Typography variant="body2" paragraph>
                  Portfolio optimization approach focused solely on minimizing risk without a specific return target.
                </Typography>
                <Link href="/docs/min-vol" passHref>
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
                  The broader theoretical framework that encompasses different portfolio optimization approaches.
                </Typography>
                <Link href="/docs/modern-portfolio-theory" passHref>
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

export default HierarchicalRiskParityPage; 