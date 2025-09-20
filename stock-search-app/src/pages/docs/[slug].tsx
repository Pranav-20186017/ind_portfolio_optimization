import { GetStaticPaths, GetStaticProps } from 'next';
import { useRouter } from 'next/router';
import SEO from '../../components/SEO';
import fs from 'fs';
import path from 'path';
import TopNav from '../../components/TopNav';
import { Box, Container, Typography, Paper, Button } from '@mui/material';
import Link from 'next/link';

// Helper function to get all doc slugs from the file system
const getDocSlugs = (): string[] => {
  const docsDirectory = path.join(process.cwd(), 'src/pages/docs');
  const filenames = fs.readdirSync(docsDirectory);
  
  // Filter out non-TSX files, index.tsx, and [slug].tsx
  const docFilenames = filenames.filter(
    filename => 
      filename.endsWith('.tsx') && 
      filename !== 'index.tsx' &&
      filename !== '[slug].tsx'
  );
  
  // Remove the .tsx extension to get the slugs
  return docFilenames.map(filename => filename.replace(/\.tsx$/, ''));
};

// Function to get a friendly title from a slug
const getDocTitle = (slug: string): string => {
  return slug
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export const getStaticPaths: GetStaticPaths = async () => {
  // For now, return empty paths to avoid conflicts with existing doc pages
  // This will make all paths fallback to the dynamic route only when they don't exist as static files
  return { 
    paths: [], 
    fallback: 'blocking' 
  };
};

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const slug = String(params?.slug || '');
  
  // Check if the doc exists
  const slugs = getDocSlugs();
  const docExists = slugs.includes(slug);
  
  if (!docExists) {
    return {
      notFound: true, // This will return a 404 page
    };
  }
  
  return { 
    props: { 
      slug,
      title: getDocTitle(slug)
    }, 
    revalidate: 86400 // Revalidate once per day
  };
};

interface DocPageProps {
  slug: string;
  title: string;
}

export default function DocPage({ slug, title }: DocPageProps) {
  const router = useRouter();
  
  // If the page is still generating via fallback, show a loading state
  if (router.isFallback) {
    return (
      <>
        <TopNav />
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Typography variant="h4">Loading...</Typography>
        </Container>
      </>
    );
  }
  
  return (
    <>
      <SEO
        title={`${title} – Indian Portfolio Optimization`}
        description={`Documentation for ${title} in the Indian Portfolio Optimization suite.`}
        path={`/docs/${slug}`}
        image="/og/docs.png"
      />
      
      <TopNav />
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Back to Docs Button */}
        <Box sx={{ mb: 4 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Documentation
            </Button>
          </Link>
        </Box>
        
        {/* The actual doc content is already in individual files, 
            this is just a wrapper with SEO for dynamic routes */}
      </Container>
    </>
  );
}
