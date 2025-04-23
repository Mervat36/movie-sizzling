const { createClient } = require('@supabase/supabase-js');

// Use your real Supabase URL and service key
const supabaseUrl = 'https://tyfttcxihduohajlzmfn.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR5ZnR0Y3hpaGR1b2hhamx6bWZuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM2OTc1MjYsImV4cCI6MjA1OTI3MzUyNn0.9H5r4RcV8ea8jTc81TVy1tZZ2rbFx9gpHKa0hfpigBk';

const supabase = createClient(supabaseUrl, supabaseKey);

module.exports = supabase;
