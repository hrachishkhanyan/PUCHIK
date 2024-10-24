#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <fstream>
#include <list>
#include <cassert>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Alpha_shape_vertex_base_3<K> Vb;
typedef CGAL::Alpha_shape_cell_base_3<K> Fb;
typedef CGAL::Triangulation_data_structure_3<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Delaunay;
typedef CGAL::Alpha_shape_3<Delaunay> Alpha_shape_3;
typedef K::Point_3 Point;
typedef Alpha_shape_3::Alpha_iterator Alpha_iterator;
typedef Alpha_shape_3::NT NT;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " input.txt" << " <alpha_value>" << " <file_suffix>" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[1]);
    NT alpha = std::strtod(argv[2], nullptr);
    std::string suffix(argv[3]);

    std::ofstream output_facets("output_facets_" + suffix + ".txt");
    std::ofstream output_cells("output_cells_" + suffix + ".txt");

    if (!input_file) {
        std::cerr << "Error: Cannot open file " << argv[1] << std::endl;
        return 1;
    }

    int n{};
    input_file >> n;
    std::cout << "Coordinates with " << n << " points" << std::endl;
//    std::list<Point> points;
    Delaunay Dt;
    std::map<Point, int> point_index_map;
    Point p;

    for (int i {}; i < n; i++) {
        input_file >> p;
        Dt.insert(p);
        point_index_map[p] = i;
    }

    Alpha_shape_3 alpha_shape(Dt);

    std::cout << "Alpha shape computed in REGULARIZED mode by default" << std::endl;
//    auto opt = alpha_shape.find_optimal_alpha(1);

//    std::cout << "Alpha shape calculated with the given value of: " << alpha << std::endl;
//    alpha_shape.set_alpha(alpha);
    if (alpha == -1) {
        auto opt = alpha_shape.find_optimal_alpha(1);
        alpha = *opt;
    }
    std::cout << alpha << std::endl;
    alpha_shape.set_alpha(alpha);
    std::cout << "Optimal alpha value to get one connected component is " << alpha << std::endl;


    for (auto fit = alpha_shape.finite_facets_begin();
    fit != alpha_shape.finite_facets_end(); ++fit) {
        if (alpha_shape.classify(*fit) == Alpha_shape_3::REGULAR) {
            auto facet = alpha_shape.triangle(*fit);
            for (auto i {0}; i < 3; i++) {
                output_facets << point_index_map[facet.vertex(i)] << " ";
            }
            output_facets << std::endl;
        }
    }
    for (auto cit = alpha_shape.finite_cells_begin();
         cit != alpha_shape.finite_cells_end(); ++cit) {
        if (alpha_shape.classify(cit) == Alpha_shape_3::INTERIOR) {
            for (auto i {0}; i < 4; i++) {
                Point pnt = cit->vertex(i)->point();
                output_cells << point_index_map[pnt] << " ";

            }
            output_cells << std::endl;
        }
    }
    output_cells.close();
    output_facets.close();
    return 0;
}