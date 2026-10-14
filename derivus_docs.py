########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

# standard imports
import os
import re
import inspect
import importlib
import logging
import json # For formatting default values
import shutil
from collections import defaultdict
from pathlib import Path
from derivus import schema


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Function to Fetch Documentation ---
def fetch_module_documentation(module_name, attribute='documentation', package='derivus'):
    """
    Fetches documentation attributes from classes within a specified module.
    """
    document_mapping = defaultdict(list)
    try:
        module = importlib.import_module(module_name, package=package)
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            # Only classes DEFINED here: `getmembers` also returns whatever this module imports,
            # so a documented class would otherwise be emitted again under every section whose
            # module imports it.
            if cls.__module__ != module.__name__:
                continue
            # A plain `Alias = Cls` assignment is the SAME class object, so the own-attribute
            # guard below cannot catch it — skip any binding whose name is not the class's own
            if class_name != cls.__name__:
                continue
            # Get 'documentation' attribute (own attribute only, not inherited via the MRO,
            # so behaviour-identical alias subclasses don't duplicate their parent's entry)
            doc = cls.__dict__.get(attribute)
            if doc and isinstance(doc, (list, tuple)) and len(doc) == 2:
                section_name, doc_content = doc
                if isinstance(doc_content, list):
                    document_mapping[section_name].append((class_name, doc_content))
                else:
                    logging.warning(f"Documentation attribute '{attribute}' in class '{class_name}' in module '{module_name}' has unexpected format.")

    except ImportError:
        logging.error(f"Could not import module '{module_name}' from package '{package}'.")
    except Exception as e:
        logging.error(f"Error fetching documentation from '{module_name}': {e}")
    return dict(document_mapping)

# --- Main Documentation Generator Class ---
class ConstructMarkdown(object):
    """
    Constructs the markdown (.md) files and mkdocs config by combining
    static content from docs_src/ and dynamic content extracted from the codebase.
    Leverages schema.mapping for detailed JSON configuration documentation.
    """
    # Insertion order IS the mkdocs nav order.
    SECTIONS = {
        'Theory': ('Theory', '.stochasticprocess', 'documentation'),
        'Valuation': ('Valuation', '.instruments', 'documentation'),
        'Bootstrapping': ('Bootstrapping', '.bootstrappers', 'documentation'),
        # each historical calibration class publishes under its own attribute, the processes
        # sharing the module with them publishing `documentation` under Theory
        'Calibration': ('Calibration', '.stochasticprocess', 'documentation_calibration'),
        'API_Usage': ('API & Usage', '.calculation', 'documentation'),
        # The hedging stack spans three modules — the JSON boundary, the bundle/environment and
        # the solver — so this section names a tuple and their class docs merge page by page.
        'Hedging': ('Hedging', ('.hedge_runtime', '.hedge_bundle', '.hedge_solver'),
                    'documentation'),
        # Developer is static-only: the attribute is one no class defines.
        'Developer': ('Developer', '.stochasticprocess', 'documentation_developer'),
        # JSON_Config names no module or attribute: `generate_json_docs` builds it from
        # `schema.mapping`, and `build` skips it here. Only the display name is read.
        'JSON_Config': ('JSON Configuration', None, None),
    }

    #: The top-level pages in nav order, `(title, file)`: a file under docs_src is copied, and one a
    #: `generate_*` method writes is listed once written; either absent, the entry is left out.
    TOP_PAGES = (('Home', 'index.md'), ('Requirements', 'require.md'),
                 ('Getting Started', 'getting_started.md'), ('Deployment', 'deployment.md'),
                 ('Jupyter Workbench', 'running_calcs.md'), ('Understanding Output', 'output.md'),
                 ('API Overview', 'api_overview.md'), ('HTTP API', 'http_api.md'),
                 ('Validating Derivus', 'validating.md'), ('Model Inventory', 'model_inventory.md'),
                 ('Model Limitations', 'limitations.md'))

    #: Where a class's `documentation` is published: its module, and the build folder its section
    #: lands in - what the inventory links each row to.
    PUBLISHED = {'.instruments': 'valuation', '.stochasticprocess': 'theory',
                 '.bootstrappers': 'bootstrapping', '.calculation': 'api_usage'}

    #: A section's pages in reading order where alphabetical is not it; a page not listed follows,
    #: sorted, and a section not listed keeps its alphabetical order with `General` first.
    READING_ORDER = {'Developer': ('Architecture', 'Calc Lifecycle', 'Dependency System',
                                   'Resolver Layer', 'Conventions', 'Market Prices',
                                   'Quote Sensitivities', 'Quote Propagation', 'Structures', 'Mcp',
                                   'Spine', 'Roadmap')}

    #: What a stated default MEANS differs by store, so every generated page opens with its own.
    STORE_NOTES = {
        'Instrument': (
            'A **Required** field must be stated: `validate()` reports one missing or blank, and the '
            'deal is never completed from the panel value shown. Every other field is a convention: '
            'omitted, it reads as its default. An undeclared key is kept in the block but read by '
            'nothing, so a misspelt convention silently takes its default.'),
        'MarketPrices': (
            "These fields sit under a block's `instrument` object; table rows are objects keyed by "
            'the column names. An omitted field reads as the same field of the family\'s '
            '`Bootstrapper Configuration` entry where that entry states it, else as the default '
            'shown - see [precedence](../bootstrapper_configuration.md#precedence).'),
        'Bootstrapper Configuration': (
            '`Prices` routes the entry to its `Market Prices` blocks; every other field is a '
            "family-wide value a block's own `instrument` overrides - see "
            '[precedence](../bootstrapper_configuration.md#precedence). The family\'s tables and '
            'containers belong to its blocks and are listed on its Market Prices page.'),
        'Calculation': (
            'A field the `Calculation` block omits reads as its default. A container the block '
            "states replaces that container's default whole and is not completed from the "
            'declarations: an omitted sub-field reads as the fallback the code consuming it applies, '
            'which is the default listed except where [Calculations](../calculations.md#containers) '
            'says otherwise. That page also gives the keys a runner adds and how overrides merge.'),
    }

    #: The generated JSON pages, one nav sub-section per store, in nav order: the `schema.mapping`
    #: path its `types` sit under, the build sub-folder, and the static page heading it as its
    #: Overview (the section takes that page's place in the nav).
    JSON_STORES = {
        'Price Factors': (('Factor',), 'price_factors', 'Price Factors Overview'),
        'Price Models': (('Process',), 'price_models', 'Price Models'),
        'Market Prices': (('MarketPrices',), 'market_prices', 'Market Prices'),
        'Bootstrapper Configuration': (('Configuration', 'Bootstrapper Configuration'),
                                       'bootstrapper_configuration', 'Bootstrapper Configuration'),
        'Trade Data': (('Instrument',), 'trade_data', 'Trade Data'),
        'Calculations': (('Calculation',), 'calculations', 'Calculations'),
    }

    #: Where a class's `documentation` lands, by the module that declares it - so a generated JSON
    #: page can link the methodology its type is priced or fitted by.
    METHODOLOGY = {'MarketPrices': ('bootstrapping', '.bootstrappers'),
                   'Bootstrapper Configuration': ('bootstrapping', '.bootstrappers'),
                   'Calculation': ('api_usage', '.calculation')}

    #: A citation as a page writes it: `[@key]`, or several keys `[@a; @b]`.
    CITATION = re.compile(r'\[(@[\w-]+(?:;\s*@[\w-]+)*)\]')

    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()
        self.docs_src_dir = self.project_dir / 'docs_src'
        self.doc_build_dir = self.project_dir / 'docs'
        self.package_name = 'derivus'
        self.static_dirs = ['stylesheets']
        # Sub‑folders inside docs_src that must be copied wholesale
        if not self.docs_src_dir.is_dir():
            logging.error(f"'docs_src' directory not found at {self.docs_src_dir}")
            raise FileNotFoundError(f"'docs_src' directory not found at {self.docs_src_dir}")
        if not schema.mapping:
            logging.error("schema.mapping is not available. JSON documentation generation will be limited.")


    def _read_static_content(self, filepath):
        """Reads content from a static markdown file."""
        try:
            with open(filepath, 'rt', encoding='utf-8') as f:
                return f.read().splitlines()
        except FileNotFoundError:
            logging.warning(f"Static file not found: {filepath}")
            return []
        except Exception as e:
            logging.error(f"Error reading static file {filepath}: {e}")
            return []

    def _generate_filename(self, page_title):
        """Generates a safe filename from a page title."""
        return page_title.lower().replace(' ', '_').replace('&', 'and').replace('/','_') + '.md'

    def _generate_nav_entry(self, title, filepath):
        """Formats a navigation entry for mkdocs.yml."""
        relative_path = filepath.relative_to(self.doc_build_dir)
        nav_path = str(relative_path).replace(os.sep, '/')
        return f"- '{title}': '{nav_path}'"

    #: A table column's token, as `schema` emits it in a row's `obj`, in words.
    COLUMN_TYPES = {'DatePicker': 'date', 'Float': 'number', 'Integer': 'integer', 'Text': 'string',
                    'Period': 'period', 'Percent': 'number, in percent', 'Basis': 'number, in basis points'}

    def _infer_data_type(self, meta, store=None):
        """Infers data type string from field metadata. `store` says how a table's rows are written,
        which differs between stores: an instrument row is an array in column order, a quote row
        an object keyed by column name."""
        widget = meta.get('widget')
        obj = meta.get('obj')
        if obj == 'Percent': return "Number (Float, in percent; see `Percent` in [General Types](../general_types.md))"
        if obj == 'Basis': return "Number (Float, in basis points; see `Basis` in [General Types](../general_types.md))"
        if widget == 'Float': return "Number (Float)"
        if widget == 'Integer': return "Number (Integer)"
        if widget == 'DatePicker': return "Date (`{\".Timestamp\": \"YYYY-MM-DD\"}`, see [General Types](../general_types.md))"
        if widget == 'Dropdown': return f"String (Allowed: `{', '.join(str(v) for v in meta.get('values', []))}`)"
        if obj == 'Period': return "String (Period, e.g., '3M', '1Y6M')"
        # Link complex types
        if widget == 'Curve' or (widget == 'Table' and obj == 'DateValueList'): return "JSON Array (See `.Curve` type in [General Types](../general_types.md))"
        if widget == 'Surface': return "JSON Object (a `.Curve` of surface points, see [General Types](../general_types.md))"
        if obj == 'DateList': return "JSON Array (See `.DateList` type in [General Types](../general_types.md))"
        if obj == 'DateEqualList': return "JSON Array (See `.DateEqualList` type in [General Types](../general_types.md))"
        if obj == 'CreditSupportList': return "JSON Array (See `.CreditSupportList` type in [General Types](../general_types.md))"
        if widget == 'Table' and meta.get('col_names'):
            if store == 'Instrument': return "JSON Array of rows, each an array in column order (columns below)"
            if store in ('MarketPrices', 'Bootstrapper Configuration'): return "JSON Array of rows, each an object keyed by column name (columns below)"
            return "JSON Array of rows (columns below)"
        if widget == 'Table': return "JSON Array of Arrays"
        if widget == 'Container': return "JSON Object (Nested Structure)"
        return "String" # Default

    def _format_default_value(self, value, meta):
        """Formats a default as inline code: a structure on one line, so it stays inside its bullet."""
        widget = meta.get('widget')
        try:
            if widget in ['Curve', 'Surface', 'Table'] and isinstance(value, str) and value != 'null':
                 # a complex type's default is a JSON string - shown as the JSON it holds
                 return f"`{json.dumps(json.loads(value))}`"
            elif value is None or value == 'null':
                 return "`null`"
            elif value == '':
                 return 'blank (`""`)'
            elif isinstance(value, (dict, list)):
                 return f"`{json.dumps(value)}`"
            else:
                 return f"`{value}`"
        except (TypeError, ValueError): # not JSON after all - shown as written
            return f"`{value}`"

    def _columns(self, meta):
        """A table's columns, `name (type)`, off the row `schema` emits; empty for a tagged table."""
        names, tokens = meta.get('col_names') or [], meta.get('obj')
        if not names:
            return ''
        tokens = tokens if isinstance(tokens, list) and len(tokens) == len(names) else [None] * len(names)
        return ', '.join('`{}`{}'.format(name, ' ({})'.format(self.COLUMN_TYPES.get(token, token)) if token else '')
                         for name, token in zip(names, tokens))

    def _field_lines(self, field_name, meta, store):
        """One top-level field as markdown: description, JSON type, what its default means, a
        table's columns and a container's fields - each nested field with the same."""
        lines = [f"### `{field_name}`",
                 f"- **Description:** {meta.get('description', field_name)}",
                 f"- **JSON Type:** {self._infer_data_type(meta, store)}"]
        lines.extend(self._value_lines(meta, '- '))
        if self._columns(meta):
            lines.append(f"- **Columns:** {self._columns(meta)}")
        if meta.get('sub_fields'):
            lines.append("- **Nested Fields:**")
            lines.extend(self._nested_lines(meta['sub_fields'], store, '  '))
        lines.append("")
        return lines

    def _value_lines(self, meta, bullet):
        """Whether a field must be stated, and its default or the value a blank panel offers."""
        lines = []
        value = meta.get('value')
        if meta.get('required'):
            lines.append(f"{bullet}**Required:** yes")
            if value not in ('', None, 'null'):
                lines.append(f"{bullet}**Panel value:** {self._format_default_value(value, meta)} "
                             "(shown on a blank panel, never applied to an omitted field)")
        else:
            lines.append(f"{bullet}**Default:** {self._format_default_value(value, meta)}")
        return lines

    def _nested_lines(self, sub_fields, store, indent):
        """A container's own fields, one bullet each, recursing into a nested container."""
        lines = []
        for name, meta in sub_fields.items():
            if not isinstance(meta, dict):
                continue
            facts = [self._infer_data_type(meta, store)]
            facts.extend(line[len('- '):] for line in self._value_lines(meta, '- '))
            if self._columns(meta):
                facts.append(f"**Columns:** {self._columns(meta)}")
            lines.append(f"{indent}- `{name}` - {meta.get('description', name).rstrip('.')}. "
                         f"{'; '.join(facts)}.")
            if meta.get('sub_fields'):
                lines.extend(self._nested_lines(meta['sub_fields'], store, indent + '  '))
        return lines

    def _methodology_link(self, store, item_type):
        """`[Methodology](...)` for the class a generated type is fitted or priced by, where that
        class carries a `documentation` attribute; '' otherwise."""
        if store not in self.METHODOLOGY:
            return ''
        section_dir, module_name = self.METHODOLOGY[store]
        module = importlib.import_module(module_name, package=self.package_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            declared = cls.__dict__.get(
                'calc_type' if store == 'Calculation' else
                'market_factor_type' if store == 'MarketPrices' else 'price_factor_type')
            doc = cls.__dict__.get('documentation')
            if declared == item_type and cls.__module__ == module.__name__ and doc:
                # the anchor mkdocs gives the `## `Class`` heading the section page writes
                return '[Methodology](../../{}/{}#{})'.format(
                    section_dir, self._generate_filename(doc[0]), name.lower())
        return ''


    def fetch_and_write_section(self, section_key, display_name, module_name, doc_attribute):
        """ Fetches dynamic docs, reads static docs, merges, and writes the output file.

        `module_name` is one module or a tuple of them; a section spanning several modules merges
        their class docs page by page. """
        logging.info(f"Processing section: {display_name}")
        nav_entries = {}
        section_src_dir = self.docs_src_dir / section_key.lower()
        section_build_dir = self.doc_build_dir / section_key.lower()
        section_build_dir.mkdir(parents=True, exist_ok=True)

        merged = defaultdict(list)
        for name in ((module_name,) if isinstance(module_name, str) else module_name):
            for page, entries in fetch_module_documentation(
                    name, doc_attribute, package=self.package_name).items():
                merged[page].extend(entries)
        module_docs = dict(merged)

        static_files = {}
        if section_src_dir.is_dir():
            for md_file in sorted(section_src_dir.glob('*.md')): # Sort for consistent order
                 page_key = md_file.stem.replace('_', ' ').title()
                 static_files[page_key] = md_file

        page_titles_ordered = list(static_files.keys())
        processed_dynamic_sections = set()

        for page_title in page_titles_ordered:
            static_filepath = static_files.get(page_title)
            final_doc = []
            if static_filepath:
                final_doc.extend(self._read_static_content(static_filepath))
                final_doc.extend(['', '---', ''])

            dynamic_content_list = module_docs.get(page_title, [])
            if dynamic_content_list:
                 processed_dynamic_sections.add(page_title)
                 for class_name, md_strings in sorted(dynamic_content_list):
                      final_doc.extend(['', f"## `{class_name}`", ''])
                      final_doc.extend(md_strings)

            build_filename = self._generate_filename(page_title)
            build_filepath = section_build_dir / build_filename
            try:
                with open(build_filepath, 'wt', encoding='utf-8') as f:
                    f.write('\n'.join(final_doc))
                nav_entries[page_title] = build_filepath
                logging.info(f"  Generated: {build_filepath}")
            except Exception as e:
                logging.error(f"  Error writing file {build_filepath}: {e}")

        # Process remaining dynamic sections
        for section_title, content_list in module_docs.items():
             if section_title not in processed_dynamic_sections and not section_title.startswith('_'):
                logging.warning(f"  Dynamic documentation found for section '{section_title}' but no corresponding static file in {section_src_dir}. Creating standalone page.")
                final_doc = [f"# {section_title}", ""]
                for class_name, md_strings in sorted(content_list):
                    final_doc.extend(['', f"## `{class_name}`", ''])
                    final_doc.extend(md_strings)

                build_filename = self._generate_filename(section_title)
                build_filepath = section_build_dir / build_filename
                try:
                    with open(build_filepath, 'wt', encoding='utf-8') as f:
                        f.write('\n'.join(final_doc))
                    nav_entries[section_title] = build_filepath
                    logging.info(f"  Generated standalone: {build_filepath}")
                except Exception as e:
                    logging.error(f"  Error writing standalone file {build_filepath}: {e}")

        return {display_name: nav_entries}


    def generate_json_docs(self):
        """Generates detailed documentation for JSON configuration using schema.mapping."""
        if not schema.mapping: return {} # Skip if the store did not assemble

        logging.info("Processing section: JSON Configuration")
        display_name = 'JSON Configuration'
        json_nav_entries = {}
        section_src_dir = self.docs_src_dir / 'json'
        section_build_dir = self.doc_build_dir / 'json'
        section_build_dir.mkdir(parents=True, exist_ok=True)

        # --- Process static overview files ---
        static_order = ['Index', 'General Types', 'System Parameters',
                        'Model Configuration', 'Price Factors Overview', 'Price Models',
                        'Correlations', 'Market Prices', 'Bootstrapper Configuration',
                        'Trade Data', 'Calculations'] # Add index.md for the section landing page
        for page_title in static_order:
            filename_base = page_title.lower().replace(' ', '_')
            static_filepath = section_src_dir / (filename_base + '.md')
            build_filepath = section_build_dir / (filename_base + '.md')
            content = self._read_static_content(static_filepath)
            if content:
                 try:
                    with open(build_filepath, 'wt', encoding='utf-8') as f:
                        f.write('\n'.join(content))
                    # Add to nav, handle 'Index' specifically
                    nav_title = 'Overview' if page_title == 'Index' else page_title
                    json_nav_entries[nav_title] = build_filepath
                    logging.info(f"  Generated JSON static: {build_filepath}")
                 except Exception as e:
                    logging.error(f"  Error writing JSON static file {build_filepath}: {e}")

        # --- Generate dynamic pages based on the declarations, one sub-section per store ---
        for section_title, (mapping_path, build_subdir, overview) in self.JSON_STORES.items():
            logging.info(f"    Generating JSON details for: {section_title}")
            mapping_data = schema.mapping
            for key in mapping_path:
                mapping_data = mapping_data.get(key, {}) if isinstance(mapping_data, dict) else {}
            if not mapping_data:
                logging.warning(f"      {' / '.join(mapping_path)} not found in schema.mapping")
                continue
            # the store a type's descriptors are read as: the bootstrapper entries are their own
            store = mapping_path[-1] if mapping_path[0] == 'Configuration' else mapping_path[0]

            details_build_dir = section_build_dir / build_subdir
            details_build_dir.mkdir(parents=True, exist_ok=True)
            section_nav = {}

            for item_type, field_identifiers in sorted(mapping_data.get('types', {}).items()):
                page_title = item_type # Use the type name as page title
                build_filename = self._generate_filename(item_type)
                build_filepath = details_build_dir / build_filename
                content = [f"# {item_type}", f"JSON configuration for the `{item_type}` type.", ""]

                # A store HOLDS its descriptors: an instrument type composes the sections that hold
                # them, a bootstrapper entry carries its aliases beside them, and a factor, process,
                # quote family or calculation type IS them
                if store == 'Instrument':
                    actual_fields = {}
                    for group_name in field_identifiers:
                        actual_fields.update(mapping_data.get('sections', {}).get(group_name, {}))
                elif store == 'Bootstrapper Configuration':
                    actual_fields = field_identifiers.get('fields', {})
                    aliases = field_identifiers.get('aliases', [])
                    content.append('An entry of the `Bootstrapper Configuration` section, keyed '
                                   f'`{item_type}`' + (' (an older book may key it '
                                   f"`{'`, `'.join(aliases)}`)" if aliases else '') + '.')
                    content.append('')
                else:
                    actual_fields = field_identifiers

                if store in self.STORE_NOTES:
                    content.extend([self.STORE_NOTES[store], ''])
                methodology = self._methodology_link(store, item_type)
                if methodology:
                    content.extend([methodology, ''])

                # Document each field
                for field_name in sorted(actual_fields):
                    content.extend(self._field_lines(field_name, actual_fields[field_name], store))

                # Write file and add to nav
                try:
                    with open(build_filepath, 'wt', encoding='utf-8') as f:
                        f.write('\n'.join(content))
                    section_nav[page_title] = build_filepath
                    logging.info(f"      Generated detail doc: {build_filepath}")
                except Exception as e:
                    logging.error(f"      Error writing detail doc file {build_filepath}: {e}")

            if not section_nav:
                continue
            # the static page heading this store becomes its Overview, closed by the list of the
            # pages generated beside it, and the store takes that page's place in the nav
            overview_path = json_nav_entries.get(overview)
            if isinstance(overview_path, Path):
                with open(overview_path, 'at', encoding='utf-8') as f:
                    f.write('\n\n## Types\n\n' + '\n'.join(
                        f"- [`{title}`]({build_subdir}/{path.name})" for title, path in section_nav.items()) + '\n')
                section_nav = dict([('Overview', overview_path)] + list(section_nav.items()))
                json_nav_entries = {(section_title if title == overview else title):
                                    (section_nav if title == overview else entry)
                                    for title, entry in json_nav_entries.items()}
            else:
                json_nav_entries[section_title] = section_nav

        return {display_name: json_nav_entries}

    def _bibliography(self):
        """`docs_src/references.md` as `{key: entry}` in file order: an entry is a `[@key]: ` line
        and the indented lines under it."""
        entries = {}
        for line in self._read_static_content(self.docs_src_dir / 'references.md'):
            match = re.match(r'\[@([\w-]+)\]:\s*(.*)', line)
            if match:
                if match[1] in entries:
                    raise ValueError(f"references.md lists [@{match[1]}] twice")
                entries[match[1]] = match[2]
            elif line.strip():
                entries[next(reversed(entries))] += ' ' + line.strip()
        return entries

    def number_references(self):
        """Numbers the bibliography in file order, writes it as the References page and turns every
        built page's `[@key]` into its linked number; a key the bibliography lacks fails the build."""
        entries = self._bibliography()
        number = {key: n for n, key in enumerate(entries, 1)}
        target = self.doc_build_dir / 'references.md'
        cited = set()
        for page in sorted(self.doc_build_dir.rglob('*.md')):
            link = Path(os.path.relpath(target, page.parent)).as_posix()

            def numbered(match):
                keys = [key.strip()[1:] for key in match[1].split(';')]
                unknown = [key for key in keys if key not in number]
                if unknown:
                    raise KeyError(f"{page.relative_to(self.doc_build_dir).as_posix()} cites "
                                   f"{', '.join(unknown)}, which docs_src/references.md does not list")
                cited.update(keys)
                return '[{}]'.format(', '.join(f'[{number[key]}]({link}#ref-{key})' for key in keys))
            text = page.read_text(encoding='utf-8')
            page.write_text(self.CITATION.sub(numbered, text), encoding='utf-8')
        for key in entries:
            if key not in cited:
                logging.warning(f"  references.md lists [@{key}], which no page cites")
        target.write_text('\n'.join(['# References', ''] + [
            f'{n}. <span id="ref-{key}"></span>{entry}' for n, (key, entry) in enumerate(entries.items(), 1)]) + '\n',
            encoding='utf-8')
        logging.info(f"  Generated: {target} ({len(entries)} works, {len(cited)} cited)")

    def build_config(self, generated_docs_nav):
        """Generates the mkdocs.yml content."""
        config_lines = [
            'site_name: Derivus',
            'site_description: Quantitative Finance library using PyTorch for pricing and risk simulation.',
            'site_author: Shuaib Osman',
            'repo_url: https://github.com/sylam/derivus', # Link to repo
            'repo_name: sylam/derivus',
            '',
            'theme:',
            '  name: material',
            '  features:',
            '    - navigation.tabs',
            '    - navigation.sections',
            '    - toc.integrate',
            '    - navigation.top',
            '    - search.suggest',
            '    - search.highlight',
            # '    - content.tabs.link', # Uncomment if using tabs in markdown
            '    - content.code.annotation',
            '    - content.code.copy',
            '  language: en',
            '  palette:',
            '    - scheme: default',
            '      toggle:',
            '        icon: material/toggle-switch-off-outline',
            '        name: Switch to dark mode',
            '      primary: teal',
            '      accent: purple',
            '    - scheme: slate',
            '      toggle:',
            '        icon: material/toggle-switch',
            '        name: Switch to light mode',
            '      primary: teal',
            '      accent: lime',
            '',
            # 14 top-level sections overflow Material's tab bar, which clips silently -
            # the stylesheet lets the bar wrap so Developer and the JSON reference stay visible
            'extra_css:',
            '  - stylesheets/extra.css',
            '',
            'extra:',
            '  social:',
            '    - icon: fontawesome/brands/github', # Correct fontawesome icon syntax
            '      link: https://github.com/sylam/derivus',
            '',
            'nav:',
        ] + [f"    - {title}: {page}" for title, page in self.TOP_PAGES
             if (self.doc_build_dir / page).is_file()]

        base_indent = "    "
        sub_indent = "        "
        sub_sub_indent = "            "
        sub_sub_sub_indent = "                " # For JSON Factor/Model/Trade types

        # Nav order IS the SECTIONS order — a parallel list here silently drops any section it
        # forgets, which is how the Developer pages went missing once already.
        for display_name, _module, _attr in self.SECTIONS.values():
            if display_name not in generated_docs_nav:
                continue
            pages = generated_docs_nav[display_name]
            config_lines.append(f"{base_indent}- {display_name}:")
            # Sort pages alphabetically within a section, a `general.md` (a chapter's index) first,
            # except where the section declares a reading order, and for JSON structure, whose
            # order is built deliberately - each store's Overview first, its types sorted after it
            order = self.READING_ORDER.get(display_name, ())
            ordered = (list if display_name == 'JSON Configuration' else
                       lambda items: sorted(items, key=lambda item: (
                           order.index(item[0]) if item[0] in order else len(order),
                           item[0] != 'General', item[0])))
            page_items = ordered(pages.items())

            for page_title, filepath_or_dict in page_items:
                if isinstance(filepath_or_dict, dict):
                    config_lines.append(f"{sub_indent}- {page_title}:")
                    for sub_page_title, sub_filepath_or_dict in ordered(filepath_or_dict.items()):
                        if isinstance(sub_filepath_or_dict, dict): # e.g., JSON -> Price Factors -> FactorType
                            config_lines.append(f"{sub_sub_indent}- {sub_page_title}:")
                            for sub_sub_title, sub_sub_filepath in sorted(sub_filepath_or_dict.items()):
                                config_lines.append(f"{sub_sub_sub_indent}{self._generate_nav_entry(sub_sub_title, sub_sub_filepath)}")
                        elif isinstance(sub_filepath_or_dict, Path):
                            config_lines.append(f"{sub_sub_indent}{self._generate_nav_entry(sub_page_title, sub_filepath_or_dict)}")
                elif isinstance(filepath_or_dict, Path):
                    config_lines.append(f"{sub_indent}{self._generate_nav_entry(page_title, filepath_or_dict)}")
        config_lines.append(f"{base_indent}- References: references.md")

        config_lines.extend([
            '',
            'extra_javascript:',
            '  - https://polyfill.io/v3/polyfill.min.js?features=es6',
            '  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
            '',
            'markdown_extensions:',
            '  - pymdownx.arithmatex:',
            '      generic: true',
            '  - pymdownx.highlight:',
            '      anchor_linenums: true',
            '  - pymdownx.inlinehilite',
            '  - pymdownx.snippets',
            '  - admonition',
            '  - pymdownx.details',
            '  - pymdownx.superfences:',
            '      custom_fences:',
            '        - name: mermaid',
            '          class: mermaid',
            '          format: !!python/name:pymdownx.superfences.fence_code_format',
            '  - pymdownx.mark',
            '  - attr_list',
            '  - md_in_html',
            '  - tables'
        ])
        return config_lines

    def generate_http_api(self):
        """`http_api.md`: every route `DV_Service` declares, with its method and summary, read off
        `derivus.service.app`. The service is the `service` extra; without it no page is written."""
        try:
            from derivus import service
        except ImportError as error:
            logging.warning(f"  No HTTP API page, the service does not import: {error}")
            return
        routes = sorted((route.path, method, (route.summary or '').replace('|', '\\|'))
                        for route in service.app.routes if getattr(route, 'include_in_schema', False)
                        for method in sorted(set(getattr(route, 'methods', ())) - {'HEAD'}))
        lines = ['# HTTP API', '',
                 "Every route `DV_Service` serves, generated from the service's own declarations. "
                 'A running service describes each in full, request and response included, at '
                 '`/docs` and `/openapi.json`; [API Overview](api_overview.md) says how the HTTP '
                 'verbs relate to the Python ones.', '',
                 '| method | path | what it does |', '| --- | --- | --- |']
        lines += [f'| `{method}` | `{path}` | {summary} |' for path, method, summary in routes]
        (self.doc_build_dir / 'http_api.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logging.info(f"  Generated: http_api.md ({len(routes)} routes)")

    def generate_inventory(self):
        """`model_inventory.md`: every instrument type, stochastic process, calibration and
        calculation the code declares, beside what prices or fits it, where its method is published
        and the test modules naming it - read off the code, so it lists what exists."""
        import ast
        from derivus import instruments, stochasticprocess, bootstrappers, calculation

        tests = {path.stem: path.read_text(encoding='utf-8')
                 for path in sorted((self.project_dir / 'tests').glob('test_*.py'))}

        def tested(name):
            found = [stem for stem, text in tests.items()
                     if f"'{name}'" in text or f'"{name}"' in text]
            shown = ', '.join(f'`{stem}`' for stem in found[:3])
            return (shown + (f' +{len(found) - 3}' if len(found) > 3 else '')) or 'none'

        def published(module, cls):
            doc = cls.__dict__.get('documentation') if cls is not None else None
            if not doc:
                return 'none'
            folder = self.PUBLISHED[module]
            stems = {p.stem.replace('_', ' ').title(): p.stem
                     for p in (self.docs_src_dir / folder).glob('*.md')}
            page = stems[doc[0]] + '.md' if doc[0] in stems else self._generate_filename(doc[0])
            return f'[{doc[0]}]({folder}/{page}#{cls.__name__.lower()})'

        # the pricing functions a deal type's own body names, else its nearest base's
        tree = ast.parse((self.project_dir / 'derivus' / 'instruments.py').read_text(encoding='utf-8'))
        bodies = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

        def pricers(name, seen=()):
            node = bodies.get(name)
            if node is None or name in seen:
                return []
            own = sorted({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                          and getattr(n.value, 'id', None) == 'pricing'
                          and n.attr.startswith(('pv_', 'pricer_'))})
            return own or next((found for base in node.bases if isinstance(base, ast.Name)
                                for found in [pricers(base.id, seen + (name,))] if found), [])

        store = schema.mapping['Instrument']
        family = {name: group.replace('New ', '') for group, names in store['groups'].items()
                  for name in names}
        rank = {group.replace('New ', ''): at for at, group in enumerate(store['groups'])}
        lines = ['# Model Inventory', '',
                 'What the library prices, simulates and fits, generated from the code: each '
                 'instrument with the pricing functions it calls, each model, each calibration and '
                 'each calculation, linked to where its method is published and to the test modules '
                 'that name it. [Validating Derivus](validating.md) says how to run those tests; '
                 '[Model Limitations](limitations.md) what is known to be wrong or limited.', '',
                 '## Instruments', '',
                 '| instrument | family | also called | priced by | method | tests |',
                 '| --- | --- | --- | --- | --- | --- |']
        for name in sorted(store['types'], key=lambda n: (rank.get(family.get(n), len(rank)), n)):
            calls = ', '.join(f'`{p}`' for p in pricers(name)) or (
                'its children' if name in store['containers'] else 'its own `generate`')
            lines.append(f"| `{name}` | {family.get(name, 'Other')} | "
                         f"{store['vernacular'].get(name, '')} | {calls} | "
                         f"{published('.instruments', getattr(instruments, name, None))} | "
                         f"{tested(name)} |")
        lines += ['', '## Models', '', '| model | simulates | method | tests |',
                  '| --- | --- | --- | --- |']
        for name in sorted(schema.mapping['Process']['types']):
            cls = getattr(stochasticprocess, name, None)
            drives = ', '.join(f'`{t}`' for t in getattr(cls, 'factor_types', ()))
            lines.append(f"| `{name}` | {drives} | {published('.stochasticprocess', cls)} | "
                         f"{tested(name)} |")
        lines += ['', '## Calibration to market prices', '',
                  '| family | quotes | writes | method | tests |', '| --- | --- | --- | --- | --- |']
        for cls in bootstrappers.FAMILIES:
            lines.append(f"| `{cls.__name__}` | `{cls.market_factor_type}` | "
                         f"`{cls.price_factor_type}` | {published('.bootstrappers', cls)} | "
                         f"{tested(cls.market_factor_type)} |")
        lines += ['', '## Calibration to history', '',
                  'Each estimator fits a model to an archive of observed prices; the methods are on '
                  '[Historical Calibration](calibration/general.md).', '',
                  '| model | estimator |', '| --- | --- |']
        estimators = {cls.__dict__['model_type']: cls.__name__
                      for cls in vars(stochasticprocess).values()
                      if isinstance(cls, type) and 'model_type' in cls.__dict__}
        lines += [f"| `{name}` | `{estimators.get(name, name)}` |"
                  for name in sorted(schema.mapping['Calibration']['types'])]
        lines += ['', '## Calculations', '', '| calculation | method | tests |',
                  '| --- | --- | --- |']
        classes = {cls.__dict__.get('calc_type', cls.__name__): cls
                   for cls in vars(calculation).values() if isinstance(cls, type)}
        for name in sorted(schema.mapping['Calculation']['types']):
            lines.append(f"| `{name}` | {published('.calculation', classes.get(name))} | "
                         f"{tested(name)} |")
        (self.doc_build_dir / 'model_inventory.md').write_text('\n'.join(lines) + '\n',
                                                              encoding='utf-8')
        logging.info('  Generated: model_inventory.md')

    def build(self):
        """Runs the documentation build process."""
        logging.info(f"Starting documentation build...")
        logging.info(f"Project Root: {self.project_dir}")
        logging.info(f"Source Docs : {self.docs_src_dir}")
        logging.info(f"Build Output: {self.doc_build_dir}")

        # a rebuild, not an overlay: a page whose type is renamed or retired is no longer generated
        # but was still sitting in the output, so the build published both names
        if self.doc_build_dir.is_dir():
            shutil.rmtree(self.doc_build_dir)
        self.doc_build_dir.mkdir(parents=True, exist_ok=True)
        generated_nav = {}

        # --- Generate Static Pages ---
        logging.info("Processing static pages...")
        for _, page in self.TOP_PAGES:
            src_path = self.docs_src_dir / page
            if not src_path.is_file():
                continue
            build_path = self.doc_build_dir / page
            content = self._read_static_content(src_path)
            if content:
                 try:
                     with open(build_path, 'wt', encoding='utf-8') as f: f.write('\n'.join(content))
                     logging.info(f"  Generated: {build_path}")
                 except Exception as e: logging.error(f"  Error writing file {build_path}: {e}")
        # --- Copy whole static directories (stylesheets/) ---

        for d in self.static_dirs:
            src_dir = self.docs_src_dir / d
            dest_dir = self.doc_build_dir / d

            if src_dir.is_dir():
                # Remove any previous artefacts for a clean copy
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                logging.info(f"  Copied directory: {src_dir}  →  {dest_dir}")
            else:
                logging.warning(f"  Static directory not found: {src_dir}")

        # --- Generate Section Docs (excluding JSON, handled separately) ---
        for section_key, (display_name, module, attr) in self.SECTIONS.items():
            if section_key != 'JSON_Config':
                nav_data = self.fetch_and_write_section(section_key, display_name, module, attr)
                generated_nav.update(nav_data)

        # --- Generate Detailed JSON Docs using the declarations ---
        json_nav_data = self.generate_json_docs()
        generated_nav.update(json_nav_data)

        # --- The two top-level pages read off the code: the HTTP verbs and the model inventory ---
        self.generate_http_api()
        self.generate_inventory()

        # --- Number the citations on every page written above, and write the References page ---
        self.number_references()

        # --- Generate mkdocs.yml ---
        logging.info("Generating mkdocs.yml...")
        mkdocs_config_content = self.build_config(generated_nav)
        mkdocs_yml_path = self.project_dir / 'mkdocs.yml'
        try:
            with open(mkdocs_yml_path, 'wt', encoding='utf-8') as f: f.write('\n'.join(mkdocs_config_content))
            logging.info(f"Generated: {mkdocs_yml_path}")
        except Exception as e: logging.error(f"Error writing {mkdocs_yml_path}: {e}")

        logging.info("Documentation build script finished.")
        print("\nTo view the documentation:")
        print(f"1. Navigate to the project directory: cd \"{self.project_dir}\"") # Added quotes for paths with spaces
        print(f"2. Run: mkdocs serve")
        print("   (Ensure mkdocs, mkdocs-material, pymdown-extensions are installed: pip install mkdocs mkdocs-material pymdown-extensions)")


# --- Main Execution Block ---
def main():
    # cwd, not __file__: installed as DV_Docs the module sits in site-packages, and docs_src
    # only ever exists in a checkout
    project_root = Path.cwd()
    logging.info(f"Using project root: {project_root}")

    if not (project_root / 'docs_src').is_dir():
        logging.error(f"No docs_src/ in {project_root}. Run DV_Docs from the repo root.")
        return 1

    ConstructMarkdown(project_root).build()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())