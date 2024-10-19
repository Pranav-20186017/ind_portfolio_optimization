// src/components/VirtualizedListbox.tsx

import React from 'react';
import { FixedSizeList, ListChildComponentProps } from 'react-window';

const LISTBOX_PADDING = 8; // Adjust based on your styling

function renderRow(props: ListChildComponentProps) {
    const { data, index, style } = props;
    return React.cloneElement(data[index], {
        style: {
            ...style,
            top: (style.top as number) + LISTBOX_PADDING,
        },
    });
}

const VirtualizedListbox = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLElement>>(
    function VirtualizedListbox(props, ref) {
        const { children, ...other } = props;
        const itemData = React.Children.toArray(children);
        const itemCount = itemData.length;
        const itemSize = 48; // Adjust based on item height

        return (
            <div ref={ref} {...other}>
                <FixedSizeList
                    height={Math.min(8, itemCount) * itemSize + 2 * LISTBOX_PADDING}
                    width="100%"
                    itemSize={itemSize}
                    itemCount={itemCount}
                    itemData={itemData}
                    overscanCount={5}
                >
                    {renderRow}
                </FixedSizeList>
            </div>
        );
    }
);

export default VirtualizedListbox;
